
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# ✅ Make sure this is the first Streamlit command
st.set_page_config(page_title="Skincare Ingredient Analyzer",
                   layout="wide",
                   page_icon="🧴")

# Sidebar Navigation Menu
with st.sidebar:
    selected = option_menu('Skincare Ingredient Analyzer',
                           ['Ingredient Analyzer',
                            'Product Recommendation'],
                           icons=['search', 'sparkels'],
                           default_index=0)

import numpy as np
import pandas as pd
import easyocr
from fuzzywuzzy import fuzz, process
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

working_dir = os.path.dirname(os.path.abspath(__file__))

if selected == "Ingredient Analyzer":
    st.title("🔍 Ingredient Analyzer")

    # Load Ingredient Model
    ingredient_model_path = os.path.join(working_dir, 'saved_models', 'ingredient_model.sav')
    ingredient_model = pickle.load(open(ingredient_model_path, 'rb'))

    # Function to analyze ingredients
    def analyze_ingredients(ingredient_list, model_data):
        harmful, warning, safe, unidentified = [], [], [], []
        ingredient_list = [i.strip().lower() for i in ingredient_list.split(',')]

        for ingredient in ingredient_list:
            match_result = process.extractOne(ingredient, model_data['name'].str.lower().tolist(), scorer=fuzz.ratio)
            if match_result:
                match, score = match_result
                if score >= 80:
                    matched_row = model_data[model_data['name'].str.lower() == match]
                    rating = matched_row['rating_num'].values[0]
                    effect = matched_row['effect'].values[0] if 'effect' in matched_row.columns else None

                    if rating == 0:
                        harmful.append((match, effect))
                    elif rating == 1:
                        warning.append((match, effect))
                    elif rating in [2, 3]:
                        safe.append(match)
                else:
                    unidentified.append(ingredient)

        return harmful, warning, safe, unidentified

    col1, col2 = st.columns(2)
    with col1:
        input_method = st.radio("Choose input method:", ["Upload Image", "Enter Ingredients Manually"])

    ingredients = ""

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
        if uploaded_file:
            from PIL import Image
            import cv2

            image = Image.open(uploaded_file)
            image_np = np.array(image)
            if len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            reader = easyocr.Reader(['en'])
            result = reader.readtext(image_np, detail=0)
            ingredients = ", ".join(result)
           # st.write("Extracted Ingredients:", ingredients)
    elif input_method == "Enter Ingredients Manually":
        ingredients = st.text_area("Enter ingredients separated by commas:")

    if st.button("Analyze Ingredients"):
        if ingredients:
            harmful, warning, safe, unidentified = analyze_ingredients(ingredients, ingredient_model)

            st.subheader("🔴 Harmful Ingredients")
            for ing, effect in harmful:
                st.write(f"- {ing} (Effect: {effect})")
            
            st.subheader("🟠 Warning Ingredients")
            for ing, effect in warning:
                st.write(f"- {ing} (Effect: {effect})")

            st.subheader("🟢 Safe Ingredients")
            st.write(", ".join(safe) if safe else "No safe ingredients found.")

            st.subheader("❓ Unidentified Ingredients")
            st.write(", ".join(unidentified) if unidentified else "All ingredients identified.")
        else:
            st.warning("Please provide ingredients for analysis.")

# ✅ Fixed incorrect condition for 'Product Recommendation'
if selected == "Product Recommendation":
    st.title("🧴 Skin Care Product Recommendation")
    file_path = os.path.join(working_dir, 'recommendation 1.xlsx')

    skin_types = ["Dry", "Normal", "Oily", "Combination", "Sensitive"]

    skin_concerns = ["Acne",
        "Acne,Dark Spots", "Acne,Oil control", "Acne,Pimples", "Acne,Refreshing",
        "Acne,Whitehead/Blackhead,Pimples", "Acne,Blackheads", "Antioxidant protection",
        "Barrier Repair,Anti aging", "Barrier Repair,Soothing", "Blackheads,Pores",
        "Brightening", "Brightening,Anti Aging", "Brightening,Even tone", "Brightening,Pigmentation",
        "Brightening,Smoothness", "Brightening,Spot reduction", "Brightening,Toning",
        "Cleansing", "Dark Spots", "Dark Spots,Pigmentation", "Dehydration", "Dullness,Radiance",
        "Exfoliation", "Exfoliation,Whitehead/Blackhead", "Glow", "Glow,Hydration", "Hydration",
        "Hydration,Anti aging", "Hydration,Broken barrier", "Hydration,Broken barrier,Irritation",
        "Hydration,Dark Spots", "Hydration,Detoxification", "Hydration,Exfoliation", "Hydration,Pore tightening",
        "Hydration,Refreshing", "Hydration,Sensitive", "Hydration,Skin barrier repair",
        "Hydration,Skin barrier strengthening", "Hydration,Skin soothing", "Hydration,Skin soothing,Irritation",
        "Hydration,Softening", "Hyperpigmentation,Skin brightening", "Intense hydration",
        "Irritation,Skin soothing", "Mild cleansing", "Moisturizing", "Moisturizing,Barrier Strengthening",
        "Moisturizing,Brightening", "Nourishment,Refreshing", "Oil control,Exfoliation",
        "Oil control,Hydration", "Pimples", "Pigmentation", "Pigmentation,Acne", "Pigmentation,Dark Spots",
        "Pigmentation,Exfoliation", "Pore Tightening,Brightening", "Pore Tightening,Oil control",
        "Pores", "Radiance,Toning", "Redness Relief", "Refreshing", "Repairing", "Sensitivity,Barrier Repair",
        "Sensitivity,Cleansing", "Sensitivity,Redness", "Skin barrier repair", "Skin barrier repair,Dehydration",
        "Skin nourishment", "Skin repair,Brightening", "Skin Repair,Anti aging", "Skin soothing,Pores",
        "Softening", "Soothing", "Soothing,Barrier Strengthening", "Soothing,Refreshing",
        "Soothing,Softening", "Sun protection", "Sun protection,Broken barrier", "Sun protection,Hydration",
        "Whitehead/Blackhead", "Whitehead/Blackhead,Exfoliation", "Wrinkle"
    ]

    product_categories = [
        "Face Cream", "Face Mask", "Face Pack", "Face Scrub", "Face Wash", "Moisturizer",
        "Night Cream", "Pigmentation Corrector", "Serum", "Sunscreen", "Toner"
    ]

    def load_and_train_model(file_path):
        df = pd.read_excel(file_path, sheet_name="skincare")
        label_encoders = {}
        for col in ["Skin type", "Concern", "Category"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        X = df[["Skin type", "Concern", "Category"]]
        y = df["Product"]
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        return rf_model, label_encoders

    def recommend_products(rf_model, label_encoders, skin_type, concern, category):
        skin_type_enc = label_encoders["Skin type"].transform([skin_type])[0]
        concern_enc = label_encoders["Concern"].transform([concern])[0]
        category_enc = label_encoders["Category"].transform([category])[0]
        input_data = np.array([[skin_type_enc, concern_enc, category_enc]])
        predictions = rf_model.predict_proba(input_data)
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        return [rf_model.classes_[i] for i in top_indices]

    model, encoders = load_and_train_model(file_path)
    st.subheader("Enter Your Skin Preferences")
    skin_type = st.selectbox("Select Your Skin Type", skin_types)
    concern = st.selectbox("Select Your Skin Concern", skin_concerns)
    category = st.selectbox("Select Product Category", product_categories)
    if st.button("Get Recommendations"):
        recommendations = recommend_products(model, encoders, skin_type, concern, category)
        st.subheader("Recommended Products for You")
        for product in recommendations:
            st.write(f"- {product}")
