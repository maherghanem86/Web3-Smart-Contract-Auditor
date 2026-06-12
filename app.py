import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

# 1. تحميل أدوات استخراج المتجهات (نعتمد CodeBERT كمعيار تقريبي للـ Space)
print("Loading tokenizers and base models...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
base_model = AutoModel.from_pretrained("microsoft/codebert-base")

def get_embeddings(code):
    # تحويل الكود إلى متجهات رقمية
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = base_model(**inputs)
    # استخدام [CLS] token كتمثيل كامل للعقد
    emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return emb

# 2. تعريف معمارية الإطار الهجين (نفس المعمارية النهائية المحسنة)
class TripleFusionSentinel(nn.Module):
    def __init__(self, input_dim=768, expert_dim=4, hidden_dim=256, num_classes=4):
        super().__init__()
        self.static_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.dynamic_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.expert_net = nn.Sequential(
            nn.Linear(expert_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        fusion_dim = (hidden_dim * 2) + 64
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.Tanh(),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, static_emb, dynamic_emb, expert_feat):
        s_feat = self.static_net(static_emb)
        d_feat = self.dynamic_net(dynamic_emb)
        e_feat = self.expert_net(expert_feat)
        combined = torch.cat((s_feat, d_feat, e_feat), dim=1)
        weights = self.attention(combined)
        fused = combined * weights
        return self.classifier(fused)

# 3. تحميل الأوزان من مستودعك مباشرة
print("Downloading the trained model weights...")
REPO_ID = "maria9659/Multi-Modal-Fusion-Security-Auditor"
FILENAME = "hybrid_fusion_results/best_fusion_model.pth"

try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = TripleFusionSentinel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# خريطة التقييمات (مرتبة أبجدياً بناءً على LabelEncoder)
labels_map = {0: "High 🔴", 1: "Low 🟡", 2: "Medium 🟠", 3: "None 🟢 (Secure)"}

# 4. دالة التوقع الرئيسية
def predict_vulnerability(code):
    if not code.strip():
        return {"Please enter some Solidity code": 1.0}

    # استخراج الميزات الأساسية
    static_emb = get_embeddings(code)
    dynamic_emb = get_embeddings(code) 
    
    # استخراج الميزات الخبيرة
    expert_feat = np.array([
        np.mean(static_emb),
        np.std(static_emb),
        np.max(dynamic_emb),
        np.linalg.norm(static_emb)
    ])

    # التحويل إلى Tensors
    t_static = torch.tensor(static_emb, dtype=torch.float32).unsqueeze(0)
    t_dynamic = torch.tensor(dynamic_emb, dtype=torch.float32).unsqueeze(0)
    t_expert = torch.tensor(expert_feat, dtype=torch.float32).unsqueeze(0)

    # التوقع (Inference)
    with torch.no_grad():
        outputs = model(t_static, t_dynamic, t_expert)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).numpy()

    # تنسيق المخرجات للواجهة
    result = {labels_map[i]: float(probabilities[i]) for i in range(4)}
    return result

# 5. تصميم واجهة Gradio
demo = gr.Interface(
    fn=predict_vulnerability,
    inputs=gr.Code(language="javascript", label="Smart Contract Code (Solidity)", lines=15),
    outputs=gr.Label(num_top_classes=4, label="Vulnerability Severity Prediction"),
    title="🛡️ Web3 Smart Contract Security Auditor (Triple Fusion Framework)",
    description="قم بلصق كود العقد الذكي (Solidity) ليقوم إطار العمل الهجين بتحليله (ثابتاً وديناميكياً) واكتشاف الأخطاء المنطقية وتصنيف خطورتها.",
    examples=[
        # مثال لعقد يحتوي على ثغرة Re-entrancy الشهيرة
        ["""pragma solidity ^0.8.0;
contract VulnerableBank {
    mapping(address => uint) public balances;
    
    function withdraw() public {
        uint bal = balances[msg.sender];
        require(bal > 0);
        (bool sent, ) = msg.sender.call{value: bal}("");
        require(sent, "Failed to send Ether");
        balances[msg.sender] = 0;
    }
}"""]
    ]
)

if __name__ == "__main__":
    demo.launch()