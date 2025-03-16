#!/usr/bin/env python
# coding: utf-8

# In[6]:


import re
import pandas as pd
import hashlib
import requests
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[7]:


# Load datasets
rockyou_path = r"C:\Users\Wasim\Downloads\rockyou.txt\rockyou.txt"
password_dataset_path = r"C:\Users\Wasim\Downloads\password_dataset.csv"
large_password_dataset_path = r"C:\Users\Wasim\Downloads\large_password_dataset.csv"


# In[8]:


# Load password datasets
df_passwords = pd.read_csv(password_dataset_path, encoding='latin1')
df_large_passwords = pd.read_csv(large_password_dataset_path, encoding='latin1')


# In[9]:


# Load RockYou dataset
with open(rockyou_path, encoding="latin1") as file:
    rockyou_passwords = file.readlines()
df_rockyou = pd.DataFrame({'password': [pw.strip() for pw in rockyou_passwords]})


# In[18]:


# Load a dataset of commonly used weak passwords
weak_passwords = ["password", "123456", "123456789", "qwerty", "abc123", "password1", "123123", "admin", "iloveyou", "welcome"]
df = pd.DataFrame(weak_passwords, columns=["Weak_Passwords"])


# In[19]:


# Sample password dataset for Machine Learning
password_data = [
    ("password123", "Weak"),
    ("Qwerty123!", "Medium"),
    ("StrongPass@2024", "Strong"),
    ("Admin1234", "Weak"),
    ("My$ecureP@ss!", "Strong"),
    ("P@ssword2023", "Medium"),
]

df_ml = pd.DataFrame(password_data, columns=["Password", "Strength"])


# In[20]:


# Feature extraction function
def extract_features(password):
    if not isinstance(password, str):  # Ensure password is a string
        return [0, 0, 0, 0, 0]  # Default empty values

    length = len(password)
    has_upper = int(bool(re.search(r'[A-Z]', password)))
    has_lower = int(bool(re.search(r'[a-z]', password)))
    has_digit = int(bool(re.search(r'\d', password)))
    has_special = int(bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)))
    return [length, has_upper, has_lower, has_digit, has_special]


# In[21]:


# Prepare features and labels
X = [extract_features(pw) for pw in df_ml["Password"] if isinstance(pw, str)]
y = df_ml["Strength"]


# In[22]:


# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# In[31]:


# Debugging: Check extracted features
print("Feature Extraction Output (X):", X[:5])  # Check first 5 samples
print("Encoded Labels (y):", y_encoded[:5])  # Check first 5 labels
print("Total Samples:", len(X))  # Ensure X has samples

if len(X) == 0:
    raise ValueError("Feature extraction failed! X is empty.")


# In[32]:


# Ensure X and y_encoded have valid data for training
if len(X) > 0 and len(y_encoded) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
else:
    raise ValueError("Training data is insufficient.")


# In[33]:


# Function to classify a password
def classify_password(password):
    features = extract_features(password)
    prediction = clf.predict([features])[0]
    return le.inverse_transform([prediction])[0]


# In[34]:


# Function to check if a password has been breached
def check_pwned_password(password):
    sha1_password = hashlib.sha1(password.encode()).hexdigest().upper()
    first5_chars = sha1_password[:5]
    response = requests.get(f"https://api.pwnedpasswords.com/range/{first5_chars}")

    if sha1_password[5:] in response.text:
        return "\u26A0\uFE0F Warning: This password has been found in data breaches!"
    return "\u2705 Your password has not been found in data breaches!"


# In[35]:


# Sample dataset for password reuse pattern analysis
password_transactions = [
    ["password123", "password@123"],
    ["qwerty", "qwerty123"],
    ["admin", "admin@2024"],
    ["iloveyou", "iloveyou123"],
    ["welcome", "welcome2023"],
]


# In[36]:


# Convert dataset for Association Rule Mining
te = TransactionEncoder()
te_array = te.fit(password_transactions).transform(password_transactions)
df_assoc = pd.DataFrame(te_array, columns=te.columns_)


# In[37]:


# Apply Apriori Algorithm
if not df_assoc.empty:
    frequent_itemsets = apriori(df_assoc, min_support=0.5, use_colnames=True)
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    else:
        rules = pd.DataFrame()
else:
    rules = pd.DataFrame()


# In[ ]:


# Input and Execution
password = input("Enter a password to check: ")
print("Predicted Strength:", classify_password(password))
print(check_pwned_password(password))
if not rules.empty:
    print("Commonly Reused Password Patterns:")
    print(rules[["antecedents", "consequents", "confidence"]])
else:
    print("No common password reuse patterns detected.")

