# 💳 Credit Card Customer Segmentation and Cluster Profiling  

This project focuses on segmenting credit card customers based on their spending, repayment, and credit usage behavior. By applying **unsupervised learning techniques**, the goal is to group customers into meaningful clusters that can help banks and financial institutions with **personalized marketing, risk management, and customer retention strategies**.  

---

## 📌 Project Overview  
- Preprocessed and scaled customer transaction data.  
- Applied **Principal Component Analysis (PCA)** for dimensionality reduction while retaining ~95% variance.  
- Implemented **K-Means clustering** to segment customers into distinct groups.  
- Conducted **cluster profiling** to interpret customer behavior and provide actionable insights.  
- Built a **Flask web application** to allow real-time prediction of customer clusters based on user input.  

---

## ⚙️ Tech Stack  
- **Python** (NumPy, Pandas, Matplotlib, Seaborn)  
- **Scikit-learn** (PCA, K-Means, preprocessing)  
- **Flask** (for deployment)  

---

## 📊 Workflow  
1. **Data Preprocessing**  
   - Handled missing values and scaled features.  
   - Engineered additional features (e.g., utilization ratios, payment ratios).  

2. **Dimensionality Reduction**  
   - Used PCA to reduce correlated features and improve clustering efficiency.  

3. **Clustering**  
   - Applied K-Means clustering and determined the optimal number of clusters using the Elbow method and Silhouette score.  

4. **Cluster Profiling**  
   - Interpreted each cluster based on customer attributes such as balance, purchases, credit limit, cash advances, and payments.  

5. **Deployment**  
   - Built a Flask web app where users can input customer details and receive their cluster assignment along with insights.  
