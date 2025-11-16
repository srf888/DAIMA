import time
import re
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any

# 配置日志系统
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("smart_finance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SmartFinanceSystem")

class UserProfile:
    """用户画像类，用于存储和管理用户信息与偏好"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.basic_info = {
            'age': None,
            'income_level': None,
            'risk_tolerance': None,  # 风险承受能力：保守/稳健/进取
            'financial_goals': [],   # 财务目标：购房/养老/教育等
            'transaction_history': [] # 交易历史
        }
        self.interaction_history = []  # 交互历史
        self.preferences = {
            'notification_frequency': 'medium',  # 通知频率：高/中/低
            'preferred_contact': 'app',          # 偏好联系方式：app/短信/邮件
            'language_style': 'formal'           # 语言风格：正式/通俗/简洁
        }
        self.fraud_risk_profile = {
            'risk_score': 0,
            'suspicious_activities': []
        }
        
    def update_basic_info(self, info: Dict[str, Any]):
        """更新用户基本信息"""
        for key, value in info.items():
            if key in self.basic_info:
                self.basic_info[key] = value
        logger.info(f"用户 {self.user_id} 基本信息已更新")
        
    def add_transaction(self, transaction: Dict[str, Any]):
        """添加交易记录"""
        self.basic_info['transaction_history'].append(transaction)
        logger.info(f"用户 {self.user_id} 添加了新交易: {transaction['transaction_id']}")
        
    def add_interaction(self, interaction: Dict[str, Any]):
        """添加交互记录"""
        self.interaction_history.append({
            **interaction,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        logger.info(f"用户 {self.user_id} 添加了新交互")
        
    def get_profile_data(self) -> Dict[str, Any]:
        """获取用户完整画像数据"""
        return {
            'user_id': self.user_id,
            'basic_info': self.basic_info,
            'preferences': self.preferences,
            'interaction_history': self.interaction_history,
            'fraud_risk_profile': self.fraud_risk_profile
        }

class AIFinancialAssistant:
    """AI金融助手核心类，实现智能服务功能"""
    
    def __init__(self):
        self.users = {}  # 用户存储：user_id -> UserProfile
        self.fraud_detector = FraudDetector()
        self.recommender = PersonalizedRecommender()
        self.conversation_manager = ConversationManager()
        self.initialize_models()
        
    def initialize_models(self):
        """初始化AI模型"""
        try:
            # 尝试加载预训练模型
            with open('fraud_model.pkl', 'rb') as f:
                self.fraud_detector.model = pickle.load(f)
            with open('recommendation_model.pkl', 'rb') as f:
                self.recommender.model = pickle.load(f)
            logger.info("预训练模型加载成功")
        except:
            # 没有预训练模型则初始化新模型
            self.fraud_detector.train_initial_model()
            self.recommender.train_initial_model()
            logger.info("初始化新模型成功")
    
    def register_user(self, user_id: str, initial_info: Dict[str, Any] = None) -> UserProfile:
        """注册新用户"""
        if user_id in self.users:
            logger.warning(f"用户 {user_id} 已存在")
            return self.users[user_id]
            
        user = UserProfile(user_id)
        if initial_info:
            user.update_basic_info(initial_info)
        self.users[user_id] = user
        logger.info(f"新用户注册: {user_id}")
        return user
    
    def process_transaction(self, user_id: str, transaction: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """处理交易并进行诈骗检测"""
        if user_id not in self.users:
            return False, {"error": "用户不存在"}
            
        user = self.users[user_id]
        user.add_transaction(transaction)
        
        # 进行诈骗风险评估
        risk_assessment = self.fraud_detector.assess_risk(user, transaction)
        
        # 如果存在风险，更新用户风险画像
        if risk_assessment['risk_level'] in ['high', 'medium']:
            user.fraud_risk_profile['risk_score'] += risk_assessment['risk_score']
            user.fraud_risk_profile['suspicious_activities'].append({
                'transaction_id': transaction['transaction_id'],
                'risk_level': risk_assessment['risk_level'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return True, {
            "transaction_status": "processed",
            "fraud_risk": risk_assessment
        }
    
    def get_personalized_recommendations(self, user_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """获取个性化金融推荐"""
        if user_id not in self.users:
            return {"error": "用户不存在"}
            
        user = self.users[user_id]
        recommendations = self.recommender.generate_recommendations(user, context)
        
        # 记录推荐交互
        user.add_interaction({
            'type': 'recommendation',
            'content': recommendations,
            'context': context
        })
        
        return recommendations
    
    def process_conversation(self, user_id: str, message: str) -> str:
        """处理用户对话，提供智能客服功能"""
        if user_id not in self.users:
            return "抱歉，未找到您的账户信息，请先注册或登录。"
            
        user = self.users[user_id]
        
        # 检测消息中是否有潜在的诈骗相关咨询
        fraud_related = self.fraud_detector.detect_fraud_related_query(message)
        if fraud_related:
            response = self.fraud_detector.generate_fraud_advice(fraud_related)
        else:
            # 生成普通对话响应
            response = self.conversation_manager.generate_response(
                message, 
                user.get_profile_data()
            )
        
        # 记录对话交互
        user.add_interaction({
            'type': 'conversation',
            'user_message': message,
            'system_response': response
        })
        
        return response
    
    def analyze_spending(self, user_id: str, period: str = "month") -> Dict[str, Any]:
        """分析用户消费情况"""
        if user_id not in self.users:
            return {"error": "用户不存在"}
            
        user = self.users[user_id]
        transactions = user.basic_info['transaction_history']
        
        if not transactions:
            return {"message": "暂无交易记录可分析"}
            
        # 转换为DataFrame进行分析
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 根据周期筛选数据
        if period == "week":
            cutoff_date = datetime.now() - timedelta(days=7)
        elif period == "month":
            cutoff_date = datetime.now() - timedelta(days=30)
        elif period == "year":
            cutoff_date = datetime.now() - timedelta(days=365)
        else:
            cutoff_date = datetime.min
            
        recent_transactions = df[df['timestamp'] >= cutoff_date]
        
        # 消费分析
        analysis = {
            'total_spent': recent_transactions['amount'].sum(),
            'transaction_count': len(recent_transactions),
            'average_transaction': recent_transactions['amount'].mean(),
            'category_breakdown': recent_transactions.groupby('category')['amount'].sum().to_dict(),
            'top_merchants': recent_transactions.groupby('merchant')['amount'].sum().nlargest(5).to_dict(),
            'spending_trend': self._calculate_trend(recent_transactions)
        }
        
        # 生成可解释的分析结果
        explanation = self._generate_spending_explanation(analysis, user.preferences['language_style'])
        
        return {
            'analysis': analysis,
            'explanation': explanation
        }
    
    def _calculate_trend(self, transactions: pd.DataFrame) -> str:
        """计算消费趋势"""
        if len(transactions) < 2:
            return "数据不足，无法判断趋势"
            
        # 按周分组计算消费
        transactions['week'] = transactions['timestamp'].dt.isocalendar().week
        weekly_spending = transactions.groupby('week')['amount'].sum()
        
        # 简单趋势判断
        if weekly_spending.iloc[-1] > weekly_spending.iloc[0]:
            return "上升"
        elif weekly_spending.iloc[-1] < weekly_spending.iloc[0]:
            return "下降"
        else:
            return "平稳"
    
    def _generate_spending_explanation(self, analysis: Dict[str, Any], language_style: str) -> str:
        """生成消费分析的自然语言解释"""
        if language_style == 'formal':
            template = (f"在过去一段时间内，您的总消费为{analysis['total_spent']:.2f}元，"
                       f"共进行了{analysis['transaction_count']}笔交易，平均每笔交易金额为{analysis['average_transaction']:.2f}元。"
                       f"消费趋势呈现{analysis['spending_trend']}态势。主要消费类别包括：{', '.join([f'{k}: {v:.2f}元' for k, v in analysis['category_breakdown'].items()])}。")
        elif language_style == 'casual':
            template = (f"最近您总共花了{analysis['total_spent']:.2f}元，一共{analysis['transaction_count']}笔交易，"
                       f"平均每笔大概{analysis['average_transaction']:.2f}元。花钱的势头在{analysis['spending_trend']}哦！"
                       f"主要花在这些地方：{', '.join([f'{k}花了{v:.2f}元' for k, v in analysis['category_breakdown'].items()])}。")
        else:  # concise
            template = (f"消费 summary：总支出{analysis['total_spent']:.2f}元，{analysis['transaction_count']}笔，"
                       f"平均{analysis['average_transaction']:.2f}元，趋势{analysis['spending_trend']}。"
                       f"主要类别：{', '.join(analysis['category_breakdown'].keys())}。")
        
        return template
    
    def export_user_data(self, user_id: str) -> str:
        """导出用户数据为JSON，用于与银行APP集成"""
        if user_id not in self.users:
            return json.dumps({"error": "用户不存在"})
            
        return json.dumps(self.users[user_id].get_profile_data(), ensure_ascii=False)

class FraudDetector:
    """金融诈骗检测系统"""
    
    def __init__(self):
        self.model = None
        self.fraud_patterns = {
            'phishing_keywords': ['验证码', '紧急', '冻结', '安全中心', '转账到安全账户', '法院', '警察', '账户异常'],
            'suspicious_amount_patterns': [
                {'min': 9999, 'max': 10001, 'reason': '接近1万元的敏感金额'},
                {'min': 49999, 'max': 50001, 'reason': '接近5万元的敏感金额'}
            ],
            'suspicious_time_patterns': [
                {'start': 0, 'end': 5, 'reason': '凌晨时段的大额交易'}
            ]
        }
    
    def train_initial_model(self):
        """初始化训练诈骗检测模型"""
        # 生成模拟训练数据
        X, y = self._generate_sample_data(10000)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # 保存模型
        with open('fraud_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
    
    def _generate_sample_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成样本数据用于模型训练"""
        # 特征：交易金额、交易时间(小时)、是否异地、是否新商户、交易频率、账户余额比例
        X = []
        y = []  # 1表示诈骗，0表示正常
        
        for _ in range(n_samples):
            amount = random.uniform(10, 100000)
            hour = random.randint(0, 23)
            is异地 = random.randint(0, 1)
            is新商户 = random.randint(0, 1)
            freq = random.uniform(0, 5)  # 最近24小时交易频率
            balance_ratio = random.uniform(0.1, 1.0)  # 交易金额/账户余额
            
            # 构造特征
            features = [amount, hour, is异地, is新商户, freq, balance_ratio]
            X.append(features)
            
            # 简单规则生成标签（实际应用中应使用真实标注数据）
            fraud_prob = 0.05  # 基础概率
            
            # 增加高风险特征的诈骗概率
            if amount > 50000:
                fraud_prob += 0.2
            if 0 <= hour < 6:
                fraud_prob += 0.15
            if is异地 and is新商户:
                fraud_prob += 0.3
            if freq > 3:
                fraud_prob += 0.15
            if balance_ratio > 0.8:
                fraud_prob += 0.2
                
            # 限制概率在0-1之间
            fraud_prob = min(1.0, max(0.0, fraud_prob))
            
            # 根据概率生成标签
            y.append(1 if random.random() < fraud_prob else 0)
            
        return np.array(X), np.array(y)
    
    def assess_risk(self, user: UserProfile, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """评估交易的诈骗风险"""
        # 提取交易特征
        amount = transaction['amount']
        timestamp = datetime.strptime(transaction['timestamp'], "%Y-%m-%d %H:%M:%S")
        hour = timestamp.hour
        
        # 判断是否异地交易
        is异地 = self._is_different_location(user, transaction.get('location', ''))
        
        # 判断是否新商户
        is新商户 = self._is_new_merchant(user, transaction.get('merchant', ''))
        
        # 计算最近交易频率
        freq = self._calculate_transaction_frequency(user)
        
        # 假设账户余额为最近交易金额的10倍（实际应从真实数据获取）
        balance = sum(t['amount'] for t in user.basic_info['transaction_history'][-10:]) if user.basic_info['transaction_history'] else amount * 5
        balance_ratio = amount / balance if balance > 0 else 1.0
        
        # 模型预测
        features = np.array([[amount, hour, is异地, is新商户, freq, balance_ratio]])
        risk_score = self.model.predict_proba(features)[0][1]  # 诈骗概率
        
        # 规则引擎补充检测
        rule_based_risk = 0
        risk_reasons = []
        
        # 检测敏感金额
        for pattern in self.fraud_patterns['suspicious_amount_patterns']:
            if pattern['min'] <= amount <= pattern['max']:
                rule_based_risk += 0.2
                risk_reasons.append(pattern['reason'])
        
        # 检测敏感时间
        for pattern in self.fraud_patterns['suspicious_time_patterns']:
            if pattern['start'] <= hour <= pattern['end'] and amount > 10000:
                rule_based_risk += 0.2
                risk_reasons.append(pattern['reason'])
        
        # 综合风险评分
        final_risk_score = min(1.0, risk_score * 0.7 + rule_based_risk * 0.3)
        
        # 确定风险等级
        if final_risk_score > 0.7:
            risk_level = 'high'
        elif final_risk_score > 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': final_risk_score,
            'risk_level': risk_level,
            'reasons': risk_reasons,
            'advice': self._generate_risk_advice(risk_level, risk_reasons)
        }
    
    def _is_different_location(self, user: UserProfile, current_location: str) -> bool:
        """判断是否为异地交易"""
        if not current_location:
            return False
            
        # 从历史交易中获取常用地点
        locations = [t.get('location', '') for t in user.basic_info['transaction_history'] if t.get('location', '')]
        if not locations:
            return False
            
        # 简单判断：与常用地点不同则视为异地
        common_locations = set(locations[-5:])  # 最近5次交易地点
        return current_location not in common_locations
    
    def _is_new_merchant(self, user: UserProfile, merchant: str) -> bool:
        """判断是否为新商户"""
        if not merchant:
            return False
            
        # 检查该商户是否在历史交易中出现过
        merchants = [t.get('merchant', '') for t in user.basic_info['transaction_history']]
        return merchant not in merchants
    
    def _calculate_transaction_frequency(self, user: UserProfile) -> float:
        """计算最近24小时的交易频率"""
        now = datetime.now()
        last_24h = 0
        
        for t in user.basic_info['transaction_history']:
            t_time = datetime.strptime(t['timestamp'], "%Y-%m-%d %H:%M:%S")
            if (now - t_time).total_seconds() < 86400:  # 24小时内
                last_24h += 1
                
        return last_24h / 24  # 每小时平均交易数
    
    def _generate_risk_advice(self, risk_level: str, reasons: List[str]) -> str:
        """生成风险提示建议"""
        if risk_level == 'high':
            base_advice = "警告：该交易存在较高诈骗风险，建议您立即停止操作。"
        elif risk_level == 'medium':
            base_advice = "注意：该交易存在一定风险，请您仔细核实。"
        else:
            return "该交易风险较低，可以正常进行。"
            
        if reasons:
            reasons_str = "主要风险点：" + "；".join(reasons)
            contact_advice = "如有疑问，请联系我们的24小时客服热线。"
            return f"{base_advice}\n{reasons_str}\n{contact_advice}"
        return base_advice
    
    def detect_fraud_related_query(self, query: str) -> Optional[str]:
        """检测用户查询中是否包含诈骗相关内容"""
        for keyword in self.fraud_patterns['phishing_keywords']:
            if keyword in query:
                return keyword
        return None
    
    def generate_fraud_advice(self, keyword: str) -> str:
        """生成针对诈骗相关查询的建议"""
        advice_map = {
            '验证码': "请注意：银行和正规机构绝不会要求您提供验证码。任何索要验证码的行为都可能是诈骗，请务必警惕！",
            '紧急': "如果接到声称有紧急情况需要转账的电话或信息，请务必通过官方渠道核实，不要轻易相信陌生人的紧急要求。",
            '冻结': "如收到账户被冻结的通知，请通过官方APP或客服热线查询，不要点击不明链接或回电陌生号码。",
            '安全中心': "官方安全中心不会通过短信、微信等方式要求您转账到所谓的'安全账户'，此类要求均为诈骗。",
            '转账到安全账户': "警方和银行绝不会设立'安全账户'，也不会要求市民转账汇款到任何所谓的'安全账户'，这是典型的诈骗手段。",
            '法院': "如收到法院传票等法律相关通知，请通过官方渠道核实，不要相信电话中要求转账销案的说法。",
            '警察': "公安机关办案有严格的程序，不会通过电话要求转账汇款，也不会在电话中制作笔录。",
            '账户异常': "如怀疑账户异常，请直接登录官方APP查询或联系客服，不要点击任何不明链接或提供账户信息。"
        }
        
        return advice_map.get(keyword, "您提到的内容可能涉及诈骗风险。请提高警惕，不要轻易向陌生账户转账，不要泄露验证码和密码。如有疑问，请联系官方客服核实。")

class PersonalizedRecommender:
    """个性化金融推荐系统"""
    
    def __init__(self):
        self.model = None
        self.products = {
            'savings': [
                {'id': 'sv1', 'name': '活期存款', 'risk': '极低', 'return': '低', 'liquidity': '高'},
                {'id': 'sv2', 'name': '定期存款', 'risk': '极低', 'return': '中低', 'liquidity': '中'}
            ],
            'investment': [
                {'id': 'iv1', 'name': '稳健型基金组合', 'risk': '低', 'return': '中', 'liquidity': '中'},
                {'id': 'iv2', 'name': '平衡型基金组合', 'risk': '中', 'return': '中高', 'liquidity': '中'},
                {'id': 'iv3', 'name': '进取型基金组合', 'risk': '高', 'return': '高', 'liquidity': '中'}
            ],
            'credit': [
                {'id': 'cd1', 'name': '消费贷', 'rate': '日息0.03%', 'max_amount': 200000},
                {'id': 'cd2', 'name': '分期卡', 'rate': '月息0.6%', 'max_amount': 100000}
            ]
        }
    
    def train_initial_model(self):
        """初始化训练推荐模型"""
        # 生成模拟训练数据
        X, y = self._generate_sample_data(5000)
        self.model = KMeans(n_clusters=5, random_state=42)  # 聚类模型，将用户分群
        self.model.fit(X)
        
        # 保存模型
        with open('recommendation_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
    
    def _generate_sample_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成样本数据用于模型训练"""
        # 特征：年龄、收入水平、风险承受能力、交易频率、平均交易金额
        X = []
        y = []  # 推荐类别标签
        
        for _ in range(n_samples):
            age = random.randint(18, 70)
            income = random.randint(1, 5)  # 1-5代表收入等级从低到高
            risk_tolerance = random.randint(1, 3)  # 1-3代表风险承受能力从低到高
            trans_freq = random.uniform(0, 10)  # 每月交易频率
            avg_amount = random.uniform(100, 10000)  # 平均交易金额
            
            X.append([age, income, risk_tolerance, trans_freq, avg_amount])
            
            # 简单规则生成推荐标签
            if risk_tolerance == 1:
                y.append(0)  # 保守型，推荐储蓄产品
            elif risk_tolerance == 2:
                y.append(1)  # 稳健型，推荐平衡产品
            else:
                y.append(2)  # 进取型，推荐投资产品
                
        return np.array(X), np.array(y)
    
    def generate_recommendations(self, user: UserProfile, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成个性化金融产品推荐"""
        # 提取用户特征
        basic_info = user.basic_info
        age = basic_info['age'] or 30
        income_level = self._map_income_to_level(basic_info['income_level'])
        risk_tolerance = self._map_risk_to_level(basic_info['risk_tolerance'])
        trans_count = len(basic_info['transaction_history'])
        trans_freq = trans_count / 30 if trans_count > 0 else 2  # 假设30天内的交易频率
        avg_amount = np.mean([t['amount'] for t in basic_info['transaction_history']]) if trans_count > 0 else 1000
        
        # 构建特征向量
        user_features = np.array([[age, income_level, risk_tolerance, trans_freq, avg_amount]])
        
        # 预测用户类别
        user_cluster = self.model.predict(user_features)[0]
        
        # 基于用户特征和上下文生成推荐
        recommendations = []
        explanations = []
        
        # 结合风险承受能力推荐产品
        if basic_info['risk_tolerance'] == '保守':
            # 优先推荐储蓄类产品
            recommendations.extend(self.products['savings'])
            explanations.append("根据您的风险偏好，为您推荐风险较低的储蓄类产品，保障资金安全。")
            
            # 补充少量低风险投资产品
            recommendations.append(self.products['investment'][0])
            explanations.append("同时为您推荐一款低风险的基金组合，可作为稳健投资的选择。")
            
        elif basic_info['risk_tolerance'] == '稳健':
            # 平衡推荐储蓄和投资产品
            recommendations.extend(self.products['savings'][1:])  # 定期存款
            recommendations.extend(self.products['investment'][:2])  # 稳健和平衡型基金
            explanations.append("根据您的风险偏好，为您推荐平衡型的资产配置方案，兼顾收益与风险。")
            
        else:  # 进取
            # 优先推荐投资类产品
            recommendations.extend(self.products['investment'][1:])  # 平衡和进取型基金
            explanations.append("根据您的风险偏好，为您推荐具有较高收益潜力的投资组合。")
        
        # 根据财务目标调整推荐
        if '购房' in basic_info['financial_goals']:
            # 推荐适合购房储备的产品
            recommendations.insert(0, self.products['savings'][1])  # 定期存款
            explanations.insert(0, "考虑到您有购房计划，优先推荐适合中长期储蓄的定期存款产品。")
        
        if '养老' in basic_info['financial_goals']:
            # 推荐适合养老规划的长期投资产品
            recommendations.insert(1, self.products['investment'][1])  # 平衡型基金
            explanations.insert(1, "为配合您的养老规划，推荐这款适合长期持有的平衡型基金组合。")
        
        # 去重并限制推荐数量
        seen_ids = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec['id'] not in seen_ids:
                seen_ids.add(rec['id'])
                unique_recommendations.append(rec)
                if len(unique_recommendations) >= 3:
                    break
        
        return {
            'recommendations': unique_recommendations,
            'explanation': " ".join(explanations),
            'personalization_factors': [
                f"风险承受能力: {basic_info['risk_tolerance']}",
                f"财务目标: {', '.join(basic_info['financial_goals']) if basic_info['financial_goals'] else '未设置'}"
            ]
        }
    
    def _map_income_to_level(self, income: Optional[str]) -> int:
        """将收入水平映射为数值等级"""
        if not income:
            return 3  # 默认中间值
            
        income_map = {
            '低收入': 1,
            '中等收入': 3,
            '高收入': 5
        }
        return income_map.get(income, 3)
    
    def _map_risk_to_level(self, risk: Optional[str]) -> int:
        """将风险承受能力映射为数值等级"""
        if not risk:
            return 2  # 默认中间值
            
        risk_map = {
            '保守': 1,
            '稳健': 2,
            '进取': 3
        }
        return risk_map.get(risk, 2)

class ConversationManager:
    """对话管理系统，处理用户自然语言交互"""
    
    def __init__(self):
        self.intent_patterns = {
            'balance_inquiry': [r'余额', r'还有多少钱', r'账户余额'],
            'transaction_history': [r'交易记录', r'消费记录', r'收支明细'],
            'bill_payment': [r'还账单', r'交账单', r'账单支付'],
            'transfer_money': [r'转账', r'转钱', r'汇款'],
            'investment_advice': [r'理财', r'投资', r'赚钱'],
            'loan_info': [r'贷款', r'借钱', r'借款'],
            'card_management': [r'卡片', r'银行卡', r'信用卡'],
            'complaint': [r'投诉', r'问题', r'错误', r'不对']
        }
    
    def generate_response(self, message: str, user_profile: Dict[str, Any]) -> str:
        """生成对话响应"""
        # 检测用户意图
        intent = self._detect_intent(message)
        
        # 根据意图生成响应，同时考虑用户偏好
        if intent == 'balance_inquiry':
            return self._generate_balance_response(user_profile)
        elif intent == 'transaction_history':
            return self._generate_transaction_response(user_profile)
        elif intent == 'bill_payment':
            return self._generate_bill_response(user_profile)
        elif intent == 'transfer_money':
            return self._generate_transfer_response(user_profile)
        elif intent == 'investment_advice':
            return self._generate_investment_response(user_profile)
        elif intent == 'loan_info':
            return self._generate_loan_response(user_profile)
        elif intent == 'card_management':
            return self._generate_card_response(user_profile)
        elif intent == 'complaint':
            return self._generate_complaint_response(user_profile)
        else:
            return self._generate_default_response(user_profile)
    
    def _detect_intent(self, message: str) -> str:
        """检测用户意图"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message):
                    return intent
        return 'unknown'
    
    def _generate_balance_response(self, user_profile: Dict[str, Any]) -> str:
        """生成余额查询响应"""
        # 模拟计算账户余额
        transactions = user_profile['basic_info']['transaction_history']
        if not transactions:
            balance = 0
        else:
            # 简单计算：假设初始余额10000，加上收入减去支出
            balance = 10000
            for t in transactions:
                if t['type'] == 'income':
                    balance += t['amount']
                else:
                    balance -= t['amount']
        
        language_style = user_profile['preferences']['language_style']
        
        if language_style == 'formal':
            return f"尊敬的客户，您当前的账户余额为{balance:.2f}元。如需了解详细收支情况，可查询交易明细。"
        elif language_style == 'casual':
            return f"你的账户里现在有{balance:.2f}元哦～ 想看钱都花在哪儿了可以看看交易记录～"
        else:
            return f"账户余额: {balance:.2f}元。可查看交易明细了解详情。"
    
    def _generate_transaction_response(self, user_profile: Dict[str, Any]) -> str:
        """生成交易记录查询响应"""
        transactions = user_profile['basic_info']['transaction_history'][-3:]  # 最近3笔
        
        if not transactions:
            return "您暂无交易记录。"
            
        trans_str = []
        for t in transactions:
            trans_type = "收入" if t['type'] == 'income' else "支出"
            trans_str.append(f"{t['timestamp']} {trans_type} {t['amount']:.2f}元，{t['merchant'] or '未知商户'}")
        
        return f"您最近的交易记录：\n" + "\n".join(trans_str) + "\n如需查看更多记录，可点击交易明细。"
    
    def _generate_bill_response(self, user_profile: Dict[str, Any]) -> str:
        """生成账单支付响应"""
        # 模拟账单信息
        has_unpaid = random.choice([True, False])
        
        if has_unpaid:
            amount = random.uniform(100, 5000)
            due_date = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
            return f"您有一笔未支付账单，金额{amount:.2f}元，到期日{due_date}。是否现在支付？"
        else:
            return "您目前没有待支付的账单。最近一期账单已结清。"
    
    def _generate_transfer_response(self, user_profile: Dict[str, Any]) -> str:
        """生成转账响应"""
        # 检查是否有常用收款人
        recipients = list(set([t.get('recipient', '') for t in user_profile['basic_info']['transaction_history'] if t.get('recipient')]))
        
        if recipients:
            return f"请输入转账金额和收款人信息。您的常用收款人：{', '.join(recipients[:3])}。"
        else:
            return "请输入转账金额和收款人信息，完成首次转账后将记录为常用收款人。"
    
    def _generate_investment_response(self, user_profile: Dict[str, Any]) -> str:
        """生成投资建议响应"""
        risk_tolerance = user_profile['basic_info']['risk_tolerance'] or '稳健'
        return f"根据您的风险偏好（{risk_tolerance}），我们为您准备了个性化的投资方案。是否查看推荐的理财产品？"
    
    def _generate_loan_response(self, user_profile: Dict[str, Any]) -> str:
        """生成贷款信息响应"""
        return "我们提供多种贷款产品，包括消费贷、经营贷等。您可以根据需求选择适合的产品，申请流程简单，审批快速。是否需要了解详细信息？"
    
    def _generate_card_response(self, user_profile: Dict[str, Any]) -> str:
        """生成卡片管理响应"""
        return "您可以在这里进行卡片挂失、密码修改、额度调整等操作。需要办理什么业务？"
    
    def _generate_complaint_response(self, user_profile: Dict[str, Any]) -> str:
        """生成投诉处理响应"""
        return "非常抱歉给您带来不好的体验。请详细说明您遇到的问题，我们会尽快为您解决。您也可以转人工客服进行一对一沟通。"
    
    def _generate_default_response(self, user_profile: Dict[str, Any]) -> str:
        """生成默认响应"""
        return "感谢您的咨询。我可以为您提供账户查询、转账汇款、投资理财等服务。请问有什么可以帮助您的？"

# 系统演示
if __name__ == "__main__":
    # 初始化智能金融系统
    finance_system = AIFinancialAssistant()
    print("智能金融系统初始化完成！")
    
    # 注册新用户
    user_id = "user_001"
    initial_info = {
        'age': 35,
        'income_level': '中等收入',
        'risk_tolerance': '稳健',
        'financial_goals': ['购房', '养老']
    }
    user = finance_system.register_user(user_id, initial_info)
    print(f"已注册用户: {user_id}")
    
    # 添加几笔交易
    transactions = [
        {
            'transaction_id': 'tx_001',
            'amount': 200,
            'type': 'expense',
            'category': '餐饮',
            'merchant': '麦当劳',
            'location': '北京市朝阳区',
            'timestamp': (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
        },
        {
            'transaction_id': 'tx_002',
            'amount': 3500,
            'type': 'expense',
            'category': '购物',
            'merchant': '京东商城',
            'location': '北京市朝阳区',
            'timestamp': (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        },
        {
            'transaction_id': 'tx_003',
            'amount': 15000,
            'type': 'income',
            'category': '工资',
            'merchant': '雇主',
            'location': '北京市朝阳区',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    ]
    
    for tx in transactions:
        success, result = finance_system.process_transaction(user_id, tx)
        if success:
            print(f"交易 {tx['transaction_id']} 处理完成，风险等级: {result['fraud_risk']['risk_level']}")
    
    # 处理一笔高风险交易
    high_risk_tx = {
        'transaction_id': 'tx_004',
        'amount': 9999,
        'type': 'expense',
        'category': '转账',
        'merchant': '未知商户',
        'location': '上海市',  # 异地
        'timestamp': (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
    }
    
    success, result = finance_system.process_transaction(user_id, high_risk_tx)
    print(f"\n高风险交易 {high_risk_tx['transaction_id']} 处理结果:")
    print(f"风险等级: {result['fraud_risk']['risk_level']}")
    print(f"风险提示: {result['fraud_risk']['advice']}")
    
    # 获取个性化推荐
    print("\n个性化金融产品推荐:")
    recommendations = finance_system.get_personalized_recommendations(user_id)
    print(f"推荐理由: {recommendations['explanation']}")
    print("推荐产品:")
    for rec in recommendations['recommendations']:
        print(f"- {rec['name']}: 风险{rec.get('risk', '')}, 收益{rec.get('return', '')}")
    
    # 消费分析
    print("\n近一个月消费分析:")
    spending_analysis = finance_system.analyze_spending(user_id, "month")
    print(spending_analysis['explanation'])
    
    # 智能客服对话
    print("\n智能客服对话:")
    queries = [
        "我的账户余额是多少？",
        "我最近有哪些交易记录？",
        "有人让我把钱转到安全账户，这是真的吗？",
        "我想了解一下理财方面的产品"
    ]
    
    for query in queries:
        print(f"用户: {query}")
        response = finance_system.process_conversation(user_id, query)
        print(f"客服: {response}\n")
    
    # 导出用户数据（用于与银行APP集成）
    user_data = finance_system.export_user_data(user_id)
    print("用户数据已导出，可用于与银行APP集成")