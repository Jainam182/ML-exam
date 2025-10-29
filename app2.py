import streamlit as st
import pandas as pd
import io
from groq import Groq
import time

# Page configuration
st.set_page_config(
    page_title="ML Lab Code Generator",
    page_icon="🤖",
    layout="wide"
)

# GROQ API KEY - Replace with your actual API key
# Example: GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY = "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"

# Verify API key is set
if not GROQ_API_KEY:
    st.error("⚠️ Please set your Groq API key in the code (line 15)")
    st.stop()

# Dictionary of all 43 ML lab questions
QUESTIONS = {
    1: "Build Linear Regression model FROM SCRATCH to predict sales w.r.t Radio features. Implement gradient descent without sklearn. Evaluate using RMSE.",
    2: "Build Linear Regression model FROM SCRATCH to predict sales w.r.t TV attribute. Implement gradient descent without sklearn. Evaluate using RMSE.",
    3: "Build Linear Regression model FROM SCRATCH to predict sales w.r.t Newspaper attribute. Implement gradient descent without sklearn. Evaluate using RMSE.",
    4: "Build Linear Regression model FROM SCRATCH to predict sales w.r.t Radio and TV. Implement gradient descent without sklearn. Evaluate using RMSE.",
    5: "Build Linear Regression model FROM SCRATCH to predict sales w.r.t Newspaper and TV. Implement gradient descent without sklearn. Evaluate using RMSE.",
    6: "Build Linear Regression model FROM SCRATCH to predict sales w.r.t Newspaper and Radio. Implement gradient descent without sklearn. Evaluate using RMSE.",
    7: "Build Regression model to predict selling prices w.r.t year_bought. Evaluate using RMSE.",
    8: "Build Regression model to predict selling prices w.r.t km_driven. Evaluate using RMSE.",
    9: "Build Regression model to predict selling prices w.r.t transmission. Evaluate using RMSE.",
    10: "Build Regression model to predict selling prices w.r.t owner. Evaluate using RMSE.",
    11: "Build Regression model to predict selling prices w.r.t year_bought and km_driven. Evaluate using RMSE.",
    12: "Build Regression model to predict selling prices w.r.t year_bought and transmission. Evaluate using RMSE.",
    13: "Build Regression model to predict selling prices w.r.t year_bought and owner. Evaluate using RMSE.",
    14: "Build Regression model to predict selling prices w.r.t year_bought and owner. Evaluate using RMSE.",
    15: "Build Regression model to predict selling prices w.r.t km_driven and transmission. Evaluate using RMSE.",
    16: "Build Regression model to predict selling prices w.r.t km_driven and owner. Evaluate using RMSE.",
    17: "Build Regression model to predict selling prices w.r.t transmission and owner. Evaluate using RMSE.",
    18: "Build Regression model to predict selling prices w.r.t transmission and owner. Evaluate using RMSE.",
    19: "Use SVM with Linear Kernel to classify whether user purchased. Create confusion matrix and evaluate using accuracy.",
    20: "Use SVM with Linear Kernel to classify whether user purchased. Create confusion matrix and evaluate using Recall.",
    21: "Use SVM with Linear Kernel to classify whether user purchased. Create confusion matrix and evaluate using Precision.",
    22: "Use SVM with RBF Kernel to classify whether user purchased. Create confusion matrix and evaluate using F1-Measure.",
    23: "Use SVM with RBF Kernel to classify whether user purchased. Create confusion matrix and evaluate using accuracy.",
    24: "Use SVM with RBF Kernel to classify whether user purchased. Create confusion matrix and evaluate using Recall.",
    25: "Use SVM with RBF Kernel to classify whether user purchased. Create confusion matrix and evaluate using Precision.",
    26: "Use SVM with RBF Kernel to classify whether user purchased. Create confusion matrix and evaluate using F1-Measure.",
    27: "Use SVM with Linear Kernel to predict iris plant category. Create confusion matrix and evaluate using Precision.",
    28: "Use SVM with Linear Kernel to predict iris plant category. Create confusion matrix and evaluate using Recall.",
    29: "Use SVM with Linear Kernel to predict iris plant category. Create confusion matrix and evaluate using Accuracy.",
    30: "Use SVM with Linear Kernel to predict iris plant category. Create confusion matrix and evaluate using F1-Measure.",
    31: "Use SVM with RBF Kernel to predict iris plant category. Create confusion matrix and evaluate using Precision.",
    32: "Use SVM with RBF Kernel to predict iris plant category. Create confusion matrix and evaluate using Recall.",
    33: "Use SVM with RBF Kernel to predict iris plant category. Create confusion matrix and evaluate using Accuracy.",
    34: "Use SVM with RBF Kernel to predict iris plant category. Create confusion matrix and evaluate using F1-Measure.",
    35: "Build decision tree classifier to predict iris plant category using GINI INDEX criteria with max_depth=4, min_samples_split=2. Evaluate using Accuracy.",
    36: "Build decision tree classifier to predict iris plant category using Entropy criteria with max_depth=4, min_samples_split=2. Evaluate using Accuracy.",
    37: "Build decision tree classifier to predict iris plant category using log loss criteria with max_depth=4, min_samples_split=2. Evaluate using Accuracy.",
    38: "Build logistic Regression classifier from scratch to predict iris plant category. Evaluate using Accuracy.",
    39: "Build Bagging Classifier model to predict iris plant category. Evaluate using Accuracy.",
    40: "Build Random Forest Classifier model to predict iris plant category. Evaluate using Accuracy.",
    41: "Build Gradient Boost Classifier model to predict iris plant category. Evaluate using Accuracy.",
    42: "Build AdaBoost Classifier model to predict iris plant category. Evaluate using Accuracy.",
    43: "Build any classifier without PCA to predict iris plant category, then apply PCA for dimensionality reduction and build same classifier. Compare the results."
}

def extract_dataset_info(uploaded_file):
    """Extract information from uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        info = {
            'columns': list(df.columns),
            'shape': df.shape,
            'rows': df.shape[0],
            'dtypes': df.dtypes.to_dict()
        }
        return df, info
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return None, None

def generate_prompt(question_idx, dataset_info):
    """Generate comprehensive prompt for Groq API"""
    question = QUESTIONS[question_idx]
    
    # Check if this requires from-scratch implementation
    from_scratch_questions = [1, 2, 3, 4, 5, 6, 38]
    is_from_scratch = question_idx in from_scratch_questions
    
    prompt = f"""You are an expert Python developer and ML engineer. Generate COMPLETE, WORKING Python code for this machine learning lab question.

**Question {question_idx}:** {question}

**Dataset Information:**
- Columns: {', '.join(dataset_info['columns'])}
- Shape: {dataset_info['shape']}
- Total Rows: {dataset_info['rows']}

"""
    
    if is_from_scratch:
        prompt += """**CRITICAL REQUIREMENT:** 
For questions 1-6 and 38, you MUST implement Linear/Logistic Regression FROM SCRATCH without using sklearn's LinearRegression or LogisticRegression classes.
- Implement gradient descent algorithm manually
- You can use numpy for matrix operations and pandas for data handling
- Do NOT use sklearn.linear_model.LinearRegression or LogisticRegression
- Show the cost function and gradient descent iterations

"""
    
    prompt += """**Code Requirements:**
1. Import all necessary libraries (numpy, pandas, matplotlib, seaborn, sklearn where appropriate)
2. Load the dataset using: df = pd.read_csv('dataset.csv')
3. Perform comprehensive data exploration and cleanup
4. Handle missing values, outliers, and data preprocessing
5. Implement the specified model/algorithm
6. Train the model and make predictions
7. Calculate and display all required metrics
8. Create visualizations (scatter plots, confusion matrix heatmaps, etc.)
9. Add detailed comments explaining each step
10. Include print statements to show intermediate results
11. Make the code ready to run in PyCharm without modifications
12. Handle categorical variables appropriately (encoding if needed)

**Output Format:**
- Provide ONLY the Python code
- No explanations outside the code
- Use clear, descriptive variable names
- Include docstrings for functions if any
- Add visualization code at the end

Generate the complete Python code now:"""
    
    return prompt

@st.cache_data(ttl=3600)
def generate_code_with_groq(question_idx, dataset_info):
    """Call Groq API to generate code"""
    try:
        # Verify API key is set
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            return "Error: API key not configured. Please set your Groq API key in the code."
        
        # Initialize Groq client with API key
        client = Groq(api_key=GROQ_API_KEY)
        prompt = generate_prompt(question_idx, dataset_info)
        
        # Use openai/gpt-oss-20b for best code generation
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python developer specializing in machine learning. Generate complete, production-ready code with proper error handling and comments."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=4500,
            top_p=0.95,
            stream=False
        )
        
        generated_code = response.choices[0].message.content
        
        # Extract code from markdown if present
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
        return generated_code
        
    except Exception as e:
        return f"Error generating code: {str(e)}\n\nPlease verify your Groq API key is correct."

# Streamlit UI
def main():
    # Header
    st.title("🤖 ML Lab Code Generator with Groq AI")
    st.markdown("Generate complete Python code for machine learning lab questions using Groq's powerful AI models")
    
    # Sidebar for information
    with st.sidebar:
        st.header("📚 About")
        st.info("""
        This app generates complete ML code for 43 lab questions:
        
        **From Scratch Implementation:**
        - Questions 1-6: Linear Regression
        - Question 38: Logistic Regression
        
        **Standard Libraries:**
        - Questions 7-37, 39-43
        
        All code is PyCharm-ready!
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Question Categories")
        st.markdown("""
        - **1-6**: Advertising Dataset (Regression)
        - **7-18**: Car Price Dataset (Regression)
        - **19-26**: Social Network (SVM Classification)
        - **27-43**: Iris Dataset (Multiple Classifiers)
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Features")
        st.markdown("""
        ✅ Complete working code
        ✅ Data preprocessing included
        ✅ Visualizations
        ✅ Model evaluation metrics
        ✅ Detailed comments
        ✅ PyCharm compatible
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Question Selection")
        question_idx = st.number_input(
            "Enter Question Number (1-43)",
            min_value=1,
            max_value=43,
            value=1,
            step=1,
            help="Select the question number for which you want to generate code"
        )
        
        # Display selected question
        if question_idx in QUESTIONS:
            st.info(f"**Question {question_idx}:** {QUESTIONS[question_idx]}")
            
            # Show special requirements for from-scratch questions
            if question_idx in [1, 2, 3, 4, 5, 6, 38]:
                st.warning("⚠️ This question requires implementation FROM SCRATCH without sklearn's built-in classes!")
    
    with col2:
        st.subheader("📂 Dataset Upload")
        uploaded_file = st.file_uploader(
            "Upload your CSV dataset",
            type=['csv'],
            help="Upload the dataset you want to use for this question"
        )
        
        if uploaded_file is not None:
            df, dataset_info = extract_dataset_info(uploaded_file)
            
            if df is not None:
                st.success(f"✅ Dataset loaded successfully!")
                
                with st.expander("📊 Dataset Preview"):
                    st.write(f"**Shape:** {dataset_info['shape']}")
                    st.write(f"**Columns:** {', '.join(dataset_info['columns'])}")
                    st.dataframe(df.head(10))
                    st.write("**Data Types:**")
                    st.write(pd.DataFrame(dataset_info['dtypes'], columns=['Type']))
    
    # Generate button
    st.markdown("---")
    
    if st.button("🚀 Generate Code", type="primary", use_container_width=True):
        # Validation
        if not uploaded_file:
            st.error("❌ Please upload a CSV dataset")
            return
        
        if not (1 <= question_idx <= 43):
            st.error("❌ Invalid question number. Please enter between 1 and 43.")
            return
        
        # Generate code
        with st.spinner("🔄 Generating code... This may take 10-30 seconds..."):
            generated_code = generate_code_with_groq(question_idx, dataset_info)
        
        if not generated_code.startswith("Error"):
            st.success("✅ Code generated successfully!")
            
            # Display generated code
            st.subheader("📄 Generated Python Code")
            st.code(generated_code, language='python', line_numbers=True)
            
            # Download button
            st.download_button(
                label="⬇️ Download Code as .py file",
                data=generated_code,
                file_name=f"ml_lab_question_{question_idx}.py",
                mime="text/x-python",
                use_container_width=True
            )
            
            # Copy to clipboard info
            st.info("💡 Tip: You can also select and copy the code directly from the code block above")
            
        else:
            st.error(generated_code)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit and Groq AI | Made for ML Lab Practicals</p>
        <p>⭐ Supports all 43 ML lab questions with intelligent code generation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()