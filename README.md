# Bank Loan Approval Patterns Analysis using NumPy

### Created by: Sangeeth M  
**Domain:** Finance / Data Analysis  

---

## 📌 Project Description  
Banks face challenges in deciding whether to approve or reject a loan application. Several factors such as **gender, income, credit score, and employment status** influence loan approvals.  

This project uses **NumPy and Pandas** to analyze loan approval patterns and correlations among applicant attributes.  

---

## 🎯 Goals
- Analyze loan approval percentages by **income, gender, and credit score**  
- Understand correlations between applicant attributes and loan approval  
- Provide insights for **data-driven decision-making**  

---

## 📊 Dataset
- **Source:** Kaggle Loan Approval Dataset (mock data generated here for demo)  
- **Sample Features:**
  - Gender (Male/Female)  
  - ApplicantIncome  
  - CreditScore  
  - LoanStatus (Approved / Rejected)  

---

## 🛠️ Implementation
- Approval % by Gender  
- Average Income (Approved vs Rejected)  
- Correlation between **Credit Score & Loan Status**  
- Correlation Matrix (Income, Credit Score, Loan Status)  

---

## 📌 Expected Output
- Approval % by Gender (e.g., Male: 75%, Female: 68%)  
- Average Income: Approved > Rejected (expected trend)  
- Correlation values showing strength of relationships  
- Correlation Matrix displaying dependencies  

---

## ✅ Conclusion
- Approval disparities can exist based on demographic/financial factors  
- **Credit Score** shows strong correlation with approval  
- **Income** also impacts approval chances  
- Gender bias may or may not be present depending on dataset  

---

## 🚀 How to Run
```bash
git clone <your-repo-url>
cd Bank_Loan_Approval_Patterns
python loan_approval_patterns.py
```

---

### 📌 Tools & Libraries
- Python 3.x  
- NumPy  
- Pandas  
