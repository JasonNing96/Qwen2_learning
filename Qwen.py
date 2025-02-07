from transformers.models.qwen2 import Qwen2Config
from transformers import AutoTokenizer
from models.Qwen2Edge import Qwen2EdgeModel
import torch
from models.benchmark import Qwen2Benchmark
from component.databases import Vectordatabase
from component.embedding import Zhipuembedding
import os
import time
import numpy as np
# 设置智谱AI的环境变量
os.environ["ZHIPUAI_API_KEY"] = "c81127e2e6644e53961ab2d0a3e8e873.UKYnxThS6jKjetkj"

def initialize_model():
    """初始化模型和tokenizer"""
    model_path = "/home/njh/LLM/Qwen2.5-1.5B-Instruct/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    config = Qwen2Config(
        vocab_size=151936,
        hidden_size=4096//2,
        intermediate_size=22016//2,
        num_hidden_layers=32//2,
        num_attention_heads=32,    
        num_key_value_heads=32,    
        max_position_embeddings=2048//2,
        rms_norm_eps=1e-6
    )
    
    model = Qwen2EdgeModel(config)
    model = model.from_pretrained(model_path)
    model = model.to("cuda").half()
    
    return model, tokenizer

def get_test_cases():
    """获取测试用例"""
    return [
        {
            "question": "请解释相对论的基本概念。",
            "reference": "相对论是爱因斯坦提出的物理学理论，包括狭义相对论和广义相对论。其核心概念是时空的相对性和能量与质量的等价性。"
        },
        # {
        #     "question": "什么是RAG？",
        #     "reference": "RAG（检索增强生成）是一种将检索系统与生成模型结合的技术，通过检索相关文档来增强语言模型的回答质量。它能提供更准确、更可靠的回答。"
        # },
        # {
        #     "question": "项目结构是怎样的？",
        #     "reference": "项目包含多个核心组件：models目录包含模型相关代码，component目录包含数据库和embedding等功能模块，主要文件包括Qwen.py等核心实现。"
        # }
    ]

def get_generation_config(tokenizer):
    """获取生成配置"""
    return {
        'max_length': 600,
        'num_return_sequences': 1,
        'do_sample': True,
        'temperature': 0.3,  # 降低温度，使输出更确定性
        'top_p': 0.85,
        'top_k': 30,  # 添加 top_k 限制
        'repetition_penalty': 1.2,  # 增加重复惩罚
        'no_repeat_ngram_size': 4,  # 增加 n-gram 重复限制
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'num_beams': 2  # 添加简单的集束搜索
    }

def run_evaluation(model, tokenizer, test_cases, is_rag=False):
    """运行评估"""
    benchmark = Qwen2Benchmark(model, tokenizer)
    metrics = {
        'f1_scores': [],
        'bleu_scores': [],
        'latency': [],
        'throughput': []
    }
    
    # 如果是RAG模式，初始化相关组件
    if is_rag:
        db = Vectordatabase()
        db.load_vector()
        embedding_model = Zhipuembedding()
    
    print(f"\n=== {'RAG' if is_rag else '基础'} 模型评估 ===")
    
    for case in test_cases:
        question = case["question"]
        reference = case["reference"]
        
        start_time = time.time()
        
        if is_rag:
            # RAG模式：先检索再生成
            context = db.query(question, embedding_model, k=2)
            context_text = "\n".join(context)
            prompt = f"""基于以下参考信息回答问题。如果参考信息不足，可以使用自己的知识补充。

参考信息：
{context_text}

问题：{question}
回答："""
        else:
            # 基础模式：直接生成
            prompt = f"请回答以下问题：{question}\n回答："
        
        # 生成回答
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs.input_ids, **get_generation_config(tokenizer))
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "回答：" in response:
            response = response.split("回答：")[-1].strip()
        
        # 计算指标
        end_time = time.time()
        latency = end_time - start_time
        num_tokens = len(tokenizer.encode(response))
        
        f1_score = benchmark.calculate_f1(response, reference)
        bleu_score = benchmark.calculate_bleu(response, reference)
        
        # 记录指标
        metrics['f1_scores'].append(f1_score)
        metrics['bleu_scores'].append(bleu_score)
        metrics['latency'].append(latency)
        metrics['throughput'].append(num_tokens / latency)
        
        # 打印单个样本结果
        print(f"\n--- 样本评估结果 ---")
        print(f"问题: {question}")
        if is_rag:
            print(f"检索到的上下文: {context_text[:200]}...")
        print(f"参考答案: {reference}")
        print(f"模型回答: {response}")
        print(f"F1分数: {f1_score:.4f}")
        print(f"BLEU分数: {bleu_score:.4f}")
        print(f"延迟: {latency:.2f}秒")
        print(f"吞吐量: {num_tokens / latency:.2f}词元/秒")
    
    # 计算平均指标
    avg_metrics = {
        'avg_f1': np.mean(metrics['f1_scores']),
        'avg_bleu': np.mean(metrics['bleu_scores']),
        'avg_latency': np.mean(metrics['latency']),
        'avg_throughput': np.mean(metrics['throughput'])
    }
    
    return avg_metrics

def run_qwen2():
    """主函数"""
    try:
        # 初始化模型和获取测试用例
        model, tokenizer = initialize_model()
        test_cases = get_test_cases()
        
        # 运行基础评估
        # 运行RAG评估
        rag_metrics = run_evaluation(model, tokenizer, test_cases, is_rag=True)
        base_metrics = run_evaluation(model, tokenizer, test_cases, is_rag=False)

        # 打印对比结果
        print("\n=== 评估对比结果 ===")
        print("指标\t\t基础模型\tRAG模型")
        print(f"F1分数\t\t{base_metrics['avg_f1']:.4f}\t{rag_metrics['avg_f1']:.4f}")
        print(f"BLEU分数\t{base_metrics['avg_bleu']:.4f}\t{rag_metrics['avg_bleu']:.4f}")
        print(f"平均延迟(秒)\t{base_metrics['avg_latency']:.2f}\t{rag_metrics['avg_latency']:.2f}")
        print(f"平均吞吐量\t{base_metrics['avg_throughput']:.2f}\t{rag_metrics['avg_throughput']:.2f}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_qwen2()   
