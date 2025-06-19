# 🧠 MathVision AI

**MathVision AI** is an advanced deep learning-powered application designed to recognize and solve handwritten mathematical expressions. It uses computer vision, LaTeX conversion, and symbolic computation—wrapped in a visually elegant interface powered by **Streamlit**.
Can be accessed at: https://huggingface.co/spaces/MSY-Salman/MathVision-AI
---

## 👨‍💻 Team Members

- **Muhammad Salman Younas** – BSDSF22M001  
- **Muhammad Hashir** – BSDSF22M017  
- **Muhammad Hassan** – BSDSF22M051

---

## 👨‍🏫 Supervisor

- **Dr. Syed Faisal Bukhari**  
  Associate Professor, Department of Data Science  
  University of the Punjab, Lahore

---

## 🚀 Tech Stack

| Layer           | Technologies Used                            |
|------------------|----------------------------------------------|
| Frontend         | [Streamlit](https://streamlit.io/)           |
| Backend          | [Python 3.x](https://www.python.org/)        |
| Deep Learning    | ONNX Runtime, Custom Trained CNN Model       |
| Computer Vision  | OpenCV, Pillow                               |
| LaTeX Processing | LaTeX2SymPy2, SymPy                          |

---

## ✨ Features

- 📷 Upload single or multiple handwritten math images  
- 🤖 Convert handwriting to LaTeX using a custom ONNX model  
- 🧠 Solve symbolic expressions using SymPy  
- 📤 Export LaTeX equations as `.tex` files  
- 📦 Download results as ZIP  
- 📊 Real-time session analytics (images processed, equations solved)  
- 🎨 Beautiful UI with custom CSS  

---

## 📂 Project Structure

    mathvision-ai/
    │
    ├── app.py              # Main Streamlit application
    ├── model.onnx          # Trained ONNX model file
    ├── keys.json           # Token-to-LaTeX mapping
    ├── requirements.txt    # Python dependencies
    └── README.md           # Project documentation

---

## ⚙️ Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/MSY-Salman/mathematical-equations-solver.git
cd mathematical-equations-solver
````

### Step 2: (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Upload image(s) containing handwritten math.
2. Image is preprocessed and passed into the ONNX model.
3. Model predicts LaTeX tokens.
4. LaTeX is converted to a symbolic expression.
5. The expression is solved using SymPy.
6. Results are rendered and downloadable.

---

## 📄 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute it for academic and educational purposes.

---
