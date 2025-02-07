from typing import List, Dict, Union
import time
import torch
import numpy as np
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import nltk
import re
import jieba
import jieba.posseg as pseg
nltk.download('punkt')
nltk.download('punkt_tab')

class Qwen2Benchmark:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # 添加中文停用词列表
        self.stopwords = {
            '的', '了', '和', '与', '或', '在', '是', '有', '等', '这', '那',
            '也', '就', '都', '而', '及', '上', '下', '到', '可以', '把', '让',
            '被', '却', '又', '及', '且', '但', '并', '很', '则', '使', '要',
            '于', '这个', '那个', '着', '给', '自己', '之', '以', '及', '了',
            '的话', '说', '对于', '这样', '那样', '如此', '这些', '那些', '什么',
            '怎么', '为什么', '如何', '哪里', '谁', '何时', '多少', '一个', '一种',
            '一样', '一般', '一直', '一定', '一些', '所以', '因此', '因为', '由于',
            '所有', '每个', '每种', '每样', '之一', '之中', '之内', '之外', '其中',
            '其他', '其它', '其实', '其余', '其次', '具有', '具体', '具体来说', '具体说来',
            '除了', '除此之外', '以及', '以至', '以至于', '以致', '因此', '因而', '进而',
            '如果', '如此', '既然', '既是', '就是', '就算', '虽然', '虽说', '尽管',
            '无论', '不管', '除非', '假如', '假使', '假若', '只要', '只有', '即使'
        }
        
    def preprocess_text(self, text: str) -> str:
        """改进的文本预处理，保留更多有意义的标点符号
        
        Args:
            text: 输入文本
        
        Returns:
            str: 预处理后的文本
        """
        # 统一标点符号
        text = re.sub(r'["""]', '"', text)  # 处理中英文引号
        text = re.sub(r'[\u2018\u2019]', "'", text)  # 使用 Unicode 编码处理单引号
        text = re.sub(r'[\u201C\u201D]', '"', text)  # 使用 Unicode 编码处理双引号
        text = re.sub(r'[（()]', '(', text)
        text = re.sub(r'[）)]', ')', text)
        
        # 保留重要的标点符号
        text = re.sub(r'[^\w\s。，！？、：；""''（）【】《》]', ' ', text)
        
        # 规范化空白字符
        text = ' '.join(text.split())
        
        # 转换为小写
        return text.lower()
    
    def extract_keywords(self, text: str) -> list:
        """提取文本中的关键词
        
        Args:
            text: 输入文本
        
        Returns:
            list: 关键词列表
        """
        # 预处理文本
        text = self.preprocess_text(text)
        
        # 分词
        words = jieba.lcut(text)
        
        # 更新停用词列表，保留更多有意义的词
        self.stopwords.difference_update({
            '是', '有', '等', '这', '那',
            '包括', '主要', '一个', '一种',
            '能够', '可以', '需要', '应该'
        })
        
        # 过滤停用词和单字词，但保留特定的单字词
        important_single_chars = {'光', '力', '能', '质', '场', '波', '核', '量', '子'}
        words = [w for w in words if (len(w) > 1 or w in important_single_chars) and w not in self.stopwords]
        
        # 只保留特定词性的词
        allowed_pos = {'n', 'v', 'a', 'eng', 'l', 'i', 'j'}  # 扩大词性范围
        words = [w for w in words if self.get_word_pos(w) in allowed_pos]
        
        # 添加词组识别
        text = ' '.join(words)
        phrases = []
        for i in range(len(words)-1):
            phrase = words[i] + words[i+1]
            if len(phrase) > 2:  # 只保留较长的词组
                phrases.append(phrase)
        
        # 合并词和词组
        keywords = words + phrases
        
        # 去重
        return list(set(keywords))
    
    def get_word_pos(self, word):
        """获取词的词性
        
        Args:
            word: 输入词
        
        Returns:
            str: 词性标记
        """
        # 使用jieba词性标注
        words = pseg.cut(word)
        for w, flag in words:
            return flag[0]
        return 'x'
    
    def calculate_f1(self, prediction, reference):
        """计算改进的F1分数
        
        Args:
            prediction: 模型预测的文本
            reference: 参考答案文本
        
        Returns:
            float: F1分数
        """
        # 提取关键词
        pred_tokens = set(self.extract_keywords(prediction))
        ref_tokens = set(self.extract_keywords(reference))
        
        # 计算部分匹配
        partial_matches = 0
        for pred in pred_tokens:
            for ref in ref_tokens:
                # 如果预测词是参考词的子串或反之
                if pred in ref or ref in pred:
                    partial_matches += 0.5
                    break
        
        # 计算完全匹配
        exact_matches = len(pred_tokens.intersection(ref_tokens))
        
        # 总匹配数
        total_matches = exact_matches + partial_matches
        
        # 避免除零错误
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        # 计算精确率和召回率
        precision = total_matches / len(pred_tokens)
        recall = total_matches / len(ref_tokens)
        
        # 避免除零错误
        if precision + recall == 0:
            return 0.0
        
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def calculate_rouge(self, prediction: str, reference: str) -> float:
        """改进的ROUGE-1分数计算
        
        Args:
            prediction: 模型预测的文本
            reference: 参考答案文本
            
        Returns:
            float: ROUGE-1分数
        """
        try:
            # 预处理文本
            def preprocess_for_rouge(text):
                """专门为ROUGE评分准备的预处理函数"""
                # 统一标点符号
                text = re.sub(r'["""]', '"', text)
                text = re.sub(r'[\u2018\u2019]', "'", text)
                text = re.sub(r'[\u201C\u201D]', '"', text)
                
                # 分句
                sentences = []
                for sent in text.split('。'):
                    if sent.strip():
                        # 使用jieba进行分词
                        words = jieba.lcut(sent)
                        # 过滤停用词但保留重要词
                        words = [w for w in words if w not in self.stopwords or len(w) > 1]
                        sentences.append(' '.join(words))
                
                return '\n'.join(sentences)
                
            # 对预测和参考文本进行预处理
            pred_text = preprocess_for_rouge(prediction)
            ref_text = preprocess_for_rouge(reference)
            
            # 计算ROUGE分数
            scores = self.rouge_scorer.score(pred_text, ref_text)
            
            # 获取ROUGE-1的F1分数
            rouge1_f1 = scores['rouge1'].fmeasure
            
            # 对分数进行调整
            if rouge1_f1 > 0:
                # 根据文本长度差异调整分数
                pred_len = len(prediction.split())
                ref_len = len(reference.split())
                length_ratio = min(pred_len, ref_len) / max(pred_len, ref_len)
                
                # 考虑关键词匹配度
                pred_keywords = set(self.extract_keywords(prediction))
                ref_keywords = set(self.extract_keywords(reference))
                keyword_overlap = len(pred_keywords & ref_keywords) / len(ref_keywords) if ref_keywords else 0
                
                # 综合考虑长度比例和关键词匹配
                adjusted_score = rouge1_f1 * (0.7 * length_ratio + 0.3 * keyword_overlap)
                return adjusted_score
                
            return rouge1_f1
            
        except Exception as e:
            print(f"ROUGE计算错误: {str(e)}")
            return 0.0
    
    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """改进的BLEU分数计算
        
        Args:
            prediction: 模型预测的文本
            reference: 参考答案文本
            
        Returns:
            float: BLEU分数
        """
        def preprocess_for_bleu(text):
            """专门为BLEU评分准备的预处理函数"""
            # 保留所有标点符号，因为它们对句子结构很重要
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r'[\u2018\u2019]', "'", text)
            text = re.sub(r'[\u201C\u201D]', '"', text)
            
            # 分词，保持标点符号
            words = []
            for sent in text.split('。'):
                if sent.strip():
                    # 使用jieba进行分词
                    words.extend(jieba.lcut(sent))
                    if sent != text.split('。')[-1]:
                        words.append('。')
                        
            # 过滤空字符但保留标点
            return [w for w in words if w.strip() or w in '。，！？、：；""''（）【】《》']
        
        try:
            # 对预测和参考文本进行预处理
            pred_tokens = preprocess_for_bleu(prediction)
            ref_tokens = [preprocess_for_bleu(reference)]  # BLEU需要参考文本是列表的列表
            
            # 如果分词结果为空，返回0分
            if not pred_tokens or not ref_tokens[0]:
                return 0.0
            
            # 根据文本长度动态调整权重
            len_pred = len(pred_tokens)
            if len_pred < 5:
                weights = (1.0, 0.0, 0.0, 0.0)  # 短文本只看unigram
            elif len_pred < 10:
                weights = (0.7, 0.3, 0.0, 0.0)  # 中等文本看unigram和bigram
            elif len_pred < 20:
                weights = (0.5, 0.3, 0.2, 0.0)  # 较长文本增加trigram
            else:
                weights = (0.4, 0.3, 0.2, 0.1)  # 长文本使用所有n-gram
            
            # 添加平滑处理
            from nltk.translate.bleu_score import SmoothingFunction
            # 使用方法7，这是一个比较全面的平滑方法
            smoothing = SmoothingFunction().method7
            
            # 计算BLEU分数
            score = sentence_bleu(
                ref_tokens, 
                pred_tokens,
                weights=weights,
                smoothing_function=smoothing
            )
            
            # 对分数进行缩放，避免过低的分数
            score = (score * 100) / 100  # 转换到0-1范围
            
            return score
            
        except Exception as e:
            print(f"BLEU计算错误: {str(e)}")
            return 0.0
    
    def generate_answer(self, question: str) -> str:
        """生成完整的回答"""
        prompt = f"请回答以下问题：{question}\n回答："
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=512,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                num_beams=1  # 添加这个参数
            )
        
        # 解码并清理回答
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 只保留回答部分
        if "回答：" in response:
            response = response.split("回答：")[-1].strip()
        elif prompt in response:
            response = response.replace(prompt, "").strip()
        return response
    
    def evaluate_batch(self, 
                      questions: List[str], 
                      references: List[str],
                      batch_size: int = 1) -> Dict[str, Union[float, Dict[str, float]]]:
        """批量评估模型性能"""
        all_metrics = {
            'f1_scores': [],
            'bleu_scores': [],
            'latency': [],
            'throughput': []
        }
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_refs = references[i:i + batch_size]
            
            for question, ref in zip(batch_questions, batch_refs):
                # 测量性能
                start_time = time.time()
                prediction = self.generate_answer(question)
                end_time = time.time()
                
                latency = end_time - start_time
                num_tokens = len(self.tokenizer.encode(prediction))
                
                # 记录性能指标
                all_metrics['latency'].append(latency)
                all_metrics['throughput'].append(num_tokens / latency)
                
                # 计算指标
                all_metrics['f1_scores'].append(self.calculate_f1(prediction, ref))
                all_metrics['bleu_scores'].append(self.calculate_bleu(prediction, ref))
                
                # 打印当前样本的评估结果
                print(f"\n--- 样本评估结果 ---")
                print(f"问题: {question}")
                print(f"参考答案: {ref}")
                print(f"模型回答: {prediction}")
                print(f"F1分数: {all_metrics['f1_scores'][-1]:.4f}")
                print(f"BLEU分数: {all_metrics['bleu_scores'][-1]:.4f}")
                print(f"延迟: {latency:.2f}秒")
                print(f"吞吐量: {num_tokens / latency:.2f}词元/秒")
        
        # 计算平均值
        return {
            'avg_f1': np.mean(all_metrics['f1_scores']),
            'avg_bleu': np.mean(all_metrics['bleu_scores']),
            'avg_latency': np.mean(all_metrics['latency']),
            'avg_throughput': np.mean(all_metrics['throughput'])
        } 