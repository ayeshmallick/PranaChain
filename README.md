# Medical MI Platform for Pranachain

<img width="1600" height="379" alt="309cff82-efe5-4843-b16c-ca48f1c5b9b7" src="https://github.com/user-attachments/assets/618d7748-bf0c-406b-a617-6fc8fc083d3a" />


<div align="center">

**ML-Powered Disease Prediction with Blockchain Security**

[![University of Windsor](https://img.shields.io/badge/University-of%20Windsor-003DA5?style=for-the-badge)](https://www.uwindsor.ca/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2-092E20?style=for-the-badge&logo=django&logoColor=white)](https://www.djangoproject.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Team](#team) • [Documentation](#documentation)

</div>

---

## **📋 Table of Contents**

- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Testing](#testing)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Team](#team)
- [License](#license)

---

## **🎯 About**

PranaChain is an innovative healthcare platform that combines **Machine Learning** and **Blockchain Technology** to revolutionize medical diagnosis and patient data management. The platform enables early detection of chronic diseases while ensuring complete patient control over their medical data through blockchain-secured access management.

### **Why PranaChain?**

Healthcare data is often siloed, insecure, and inaccessible to those who need it most—patients and practitioners. PranaChain addresses these challenges by:

✅ **Patient Empowerment** - Patients maintain full ownership and control of their health records  
✅ **ML-Driven Insights** - Early detection of Chronic Kidney Disease (CKD), Diabetes, and Heart Disease  
✅ **Blockchain Security** - Immutable, transparent audit trail for data access and predictions  
✅ **Explainable ML** - SHAP visualizations help doctors understand model decisions  
✅ **Seamless Integration** - OCR-powered blood report analysis for automated indicator extraction

---

## **✨ Features**

### **For Patients**

- **Upload Blood Reports** - Simple drag-and-drop interface for PDF/image uploads
- **Automated OCR** - Intelligent extraction of medical indicators from reports
- **Multi-Disease Predictions** - Get predictions for CKD, Diabetes, and Heart Disease simultaneously
- **Visual Risk Assessment** - Color-coded risk levels (Low, Medium, High, Critical)
- **Blockchain Access Control** - Grant/revoke doctor access with one click
- **Mobile Responsive** - Access your predictions from any device

### **For Doctors**

- **Patient Dashboard** - View all accessible patient predictions in one place
- **SHAP Explainability** - Understand which factors drive each prediction with AI visualizations
- **Clinical Notes** - Add professional assessments and treatment recommendations
- **Analytics** - View aggregate statistics and patient trends
- **Detailed Analysis** - Access complete medical indicators and reference ranges

### **Core Technology**

- **Machine Learning Models**:
  - **CKD**: Random Forest Ensemble (95% accuracy)
  - **Diabetes**: Random Forest Classifier (92% accuracy)
  - **Heart Disease**: K-Nearest Neighbors (89% accuracy)
- 🔗 **Blockchain**: Custom Python implementation with immutable access logs
- **SHAP**: SHapley Additive exPlanations for model interpretability
- **OCR**: Tesseract-based medical indicator extraction

---

## **🛠 Tech Stack**

### **Backend**
- **Framework**: Django 4.2 + Django REST Framework
- **Language**: Python 3.9+
- **ML Libraries**: scikit-learn, SHAP, NumPy, Pandas
- **OCR**: Tesseract OCR, pdf2image
- **Database**: PostgreSQL / SQLite

### **Frontend**
- **Framework**: React 18
- **Styling**: Tailwind CSS
- **State Management**: React Context API
- **Charts**: Recharts
- **HTTP Client**: Axios

### **DevOps & Tools**
- **Version Control**: Git & GitHub
- **Package Management**: pip (Python), npm (JavaScript)
- **Environment**: Python venv
- **API Testing**: Postman

---

## **🏗 System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Patient    │  │    Doctor    │  │    Admin     │       │
│  │  Dashboard   │  │  Dashboard   │  │   Panel      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Django REST API Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │     Auth     │  │  Predictions │  │   Reports    │       │
│  │  Endpoints   │  │  Endpoints   │  │  Endpoints   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  OCR Engine  │  │  ML Models   │  │  Blockchain  │       │
│  │  (Tesseract) │  │   (SHAP)     │  │   Ledger     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer (PostgreSQL)                │
│       Users | Reports | Predictions | Blockchain Logs       │
└─────────────────────────────────────────────────────────────┘
```

---

## **⚙️ Installation**

### **Prerequisites**

Ensure you have the following installed:

- **Python** 3.9 or higher
- **Node.js** 16+ and npm
- **PostgreSQL** (optional, SQLite works for development)
- **Tesseract OCR** ([Installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html))
- **Git**

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/your-username/pranachain.git
cd pranachain
```

### **Step 2: Backend Setup**

#### **2.1 Create Virtual Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### **2.2 Install Python Dependencies**

```bash
pip install -r requirements.txt
```

#### **2.3 Configure Environment Variables**

Create a `.env` file in the root directory:

```env
# Django Settings
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database (PostgreSQL)
DATABASE_URL=postgresql://username:password@localhost:5432/pranachain

# Or use SQLite for development
# DATABASE_URL=sqlite:///db.sqlite3

# ML Models Path
ML_MODEL_PATH=ml_models/ckd_model.pkl

# Media Files
MEDIA_ROOT=media/
MEDIA_URL=/media/

# Tesseract OCR Path (adjust for your OS)
TESSERACT_CMD=/usr/bin/tesseract  # Linux/Mac
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe  # Windows
```

#### **2.4 Run Migrations**

```bash
python manage.py makemigrations
python manage.py migrate
```

#### **2.5 Create Superuser (Admin)**

```bash
python manage.py createsuperuser
```

#### **2.6 Load ML Models**

Ensure ML model files are in the `ml_models/` directory:
- `ckd_model.pkl`
- `diabetes_model.pkl`
- `heart_knn_model.pkl`
- `heart_dt_model.pkl`
- `transformer.pkl`
- `scaler.pkl`

#### **2.7 Start Django Server**

```bash
python manage.py runserver
```

The backend API will be available at `http://localhost:8000`

---

### **Step 3: Frontend Setup**

#### **3.1 Navigate to Frontend Directory**

```bash
cd frontend
```

#### **3.2 Install Node Dependencies**

```bash
npm install
```

#### **3.3 Configure API URL**

Edit `frontend/src/config.js`:

```javascript
export const API_BASE_URL = 'http://localhost:8000/api';
```

#### **3.4 Start React Development Server**

```bash
npm start
```

The frontend will be available at `http://localhost:3000`

---

### **Step 4: Verify Installation**

1. Open browser and navigate to `http://localhost:3000`
2. Register a new patient account
3. Upload a sample blood report
4. Verify predictions are generated
5. Test blockchain access control

---

## **🧪 Testing**

### **Backend Testing**

#### **Run All Tests**

```bash
python manage.py test
```

#### **Test Specific Apps**

```bash
# Test authentication
python manage.py test accounts

# Test predictions
python manage.py test predictions

# Test OCR and reports
python manage.py test reports
```

#### **Test Coverage**

```bash
# Install coverage
pip install coverage

# Run tests with coverage
coverage run --source='.' manage.py test

# Generate coverage report
coverage report
coverage html  # Creates htmlcov/index.html
```

### **Frontend Testing**

```bash
cd frontend

# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

### **API Endpoint Testing**

Use the provided Postman collection or test manually:

#### **Example: Register Patient**

```bash
curl -X POST http://localhost:8000/api/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "patient@test.com",
    "password": "testpass123",
    "first_name": "John",
    "last_name": "Doe",
    "user_type": "patient",
    "age": 35,
    "bmi": 24.5
  }'
```

#### **Example: Upload Blood Report**

```bash
curl -X POST http://localhost:8000/api/reports/upload/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "report_file=@sample_blood_report.pdf" \
  -F "report_title=Annual Checkup" \
  -F "test_date=2024-11-20"
```

### **Testing Checklist**

- [ ] User registration (Patient & Doctor)
- [ ] User login and JWT authentication
- [ ] Blood report upload and OCR extraction
- [ ] CKD prediction with SHAP charts
- [ ] Diabetes prediction with SHAP charts
- [ ] Heart Disease prediction with feature charts
- [ ] Grant access to doctor (Blockchain)
- [ ] Revoke access from doctor (Blockchain)
- [ ] Doctor viewing patient predictions
- [ ] Doctor adding clinical notes
- [ ] Profile updates
- [ ] Report deletion

---

## **📖 Usage**

### **Quick Start Guide**

#### **As a Patient:**

1. **Register**: Create an account with your health profile
2. **Upload Report**: Upload your blood test report (PDF/Image)
3. **View Predictions**: Get instant predictions for 3 diseases
4. **Share with Doctor**: Grant access to your doctor via blockchain
5. **Track Progress**: Monitor your health trends over time

#### **As a Doctor:**

1. **Register**: Create a doctor account with medical license
2. **Access Patients**: View predictions patients have shared with you
3. **Analyze Results**: Review SHAP charts and medical indicators
4. **Add Notes**: Provide clinical assessments and recommendations
5. **Track Patients**: Monitor multiple patients from one dashboard

### **Detailed Documentation**

For comprehensive usage instructions, refer to the [User Manual](docs/USER_MANUAL.md).

---

## **📸 Screenshots**

### **Patient Dashboard**
<img width="1147" height="994" alt="Screenshot 2025-11-20 at 9 24 53 AM" src="https://github.com/user-attachments/assets/48daa245-1f88-4f59-abf9-aabdc4659bb8" />

*Upload blood reports and view predictions with color-coded risk levels*

### **SHAP Explainability (CKD)**
<img width="540" height="432" alt="image" src="https://github.com/user-attachments/assets/b1e65459-29b0-409e-b7a2-2781111745b8" />

*SHAP waterfall chart showing which factors drive the prediction*

### **Blockchain Access Control**
<img width="968" height="532" alt="Screenshot 2025-11-20 at 9 25 22 AM" src="https://github.com/user-attachments/assets/b6d93b5c-ae9f-4ba9-8601-3edd0d3817c7" />

*Grant or revoke doctor access with blockchain security*

### **Doctor Dashboard**
<img width="1501" height="764" alt="Screenshot 2025-11-20 at 9 29 31 AM" src="https://github.com/user-attachments/assets/84b28a5a-04a8-49e6-ae16-92316a15892d" />

*View all accessible patient predictions with analytics*

---

## **👥 Team**

<div align="center">

### **Team R.O.B.O.T.S - Group 16**

<img width="1226" height="698" alt="516617339-12243a67-555b-4527-bacc-58a7ef04d690" src="https://github.com/user-attachments/assets/a3f6f28a-1a82-4c98-9785-34536e6b6ffe" />


</div>

| Name | Role | Responsibilities | Student ID |
|------|------|------------------|------------|
| **Rithish Ashwin Suresh Kumar** | Team Lead & Blockchain Architect | Blockchain security, Agile management, Backend integration | 110182109 |
| **Shaurya Parshad** | ML Scientist & SDE | UI design, Backend development, CKD prediction model | 110191553 |
| **Timothy Eric Dare James Dare** | ML Engineer & Data Scientist | Data preprocessing, Model training | 110191780 |
| **Sandeep Kapoor** | ML Engineer & Data Scientist | Diabetes & Heart Disease models, Feature engineering | 110184820 |
| **Ayeshkant Mallick** | ML Engineer & Algorithm Specialist | Algorithm optimization, SHAP implementation | 110190414 |

**Academic Institution**: University of Windsor  
**Course**: COMP8967-1-R-2025F|Internship Project I  
**Semester**: Fall 2025

## **🔒 Security & Privacy**

PranaChain implements multiple layers of security:

- **🔐 JWT Authentication** - Secure token-based authentication
- **🔗 Blockchain Access Control** - Immutable audit trail for data access
- **🔒 HTTPS Encryption** - All data transmitted securely (in production)
- **🛡️ HIPAA Compliance** - Following healthcare data protection standards
- **👤 User Consent** - Patients control all data sharing
- **📝 Audit Logs** - Complete history of who accessed what data

---

## **⚠️ Disclaimer**

**Important Medical Notice:**

PranaChain is an **educational project** and provides **screening predictions only**, not medical diagnoses. The AI models are designed as decision support tools for healthcare professionals.

**Do NOT:**
- Use predictions as the sole basis for medical decisions
- Self-diagnose or self-treat based on AI results
- Delay seeking professional medical advice
- Ignore symptoms based on low-risk predictions

**Always:**
- Consult a licensed physician for medical evaluation
- Get proper diagnostic tests ordered by your doctor
- Follow professional medical guidance
- Report concerning symptoms immediately

The platform creators and contributors are not responsible for any medical decisions made using this system.

---

## **📄 License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **🤝 Contributing**

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### **Development Workflow**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## **📞 Support & Contact**

- **Issues**: [GitHub Issues](https://github.com/your-username/pranachain/issues)
- **Email**: parshad@uwindsor.ca

---

## **🙏 Acknowledgments**

- University of Windsor for academic support
- Healthcare professionals who provided domain expertise
- Open-source ML community for libraries and tools
- Dataset providers: UCI Machine Learning Repository, Kaggle

---

## **📊 Project Status**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-87%25-green)
![Version](https://img.shields.io/badge/version-1.0.0-blue)

**Current Phase**: Production Ready ✅

**Recent Updates**:
- ✅ Multi-disease prediction support
- ✅ SHAP explainability for CKD & Diabetes
- ✅ Blockchain access control
- ✅ Mobile-responsive UI
- ✅ Clinical notes feature

**Roadmap**:
- 🚧 Additional disease models (Liver disease, Thyroid)
- 🚧 Real-time notifications
- 🚧 Multi-language support
- 🚧 Mobile app (iOS/Android)

---

<div align="center">

**✨ PranaChain - Empowering Patients, Enabling Doctors, Advancing Healthcare ✨**


[⬆ Back to Top](#pranachain-medical-ai-platform)

</div>
