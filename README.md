
### Application flow:

- Select between **Ingredient Analyzer** or **Product Recommendation** in the sidebar.
- For ingredient analysis, either upload an image of the skincare product ingredient label or manually enter the ingredient list.
- Click on **Analyze Ingredients** to see categorized ingredient safety details.
- For product recommendations, select your skin type, skin concern, and desired product category, then hit **Get Recommendations**.

---

## Project Structure

skincare-ingredient-analyzer/
│
├── saved_models/               # Pre-trained ML model files
├── data/
│   ├── ingredientanalyzer.csv # Ingredient safety dataset
│   └── recommendation_1.xlsx  # Skin product recommendation dataset
├── pro.py                     # Streamlit main app script
├── skincare_ingredient_analyzer.py  # Backend code for ingredient analysis
├── README.md                  # Project documentation and overview
├── requirements.txt           # Python dependencies list
├── docs/                      # Project report and additional documentation

---

## Datasets and Models

- **Ingredient Data:** Contains ingredient names, safety ratings, and effects.
- **Product Recommendation Data:** Details on skin types, skin concerns, product categories, and recommended products.
- **Models:** Random Forest classifier trained on product dataset, and ingredient safety data loaded for ingredient classification.

---

## Testing

- Unit tests cover:
  - OCR text extraction accuracy
  - Ingredient classification via fuzzy matching
  - Recommendation system predictions
- Integration tests ensure a smooth workflow from input to output.
- Tested on various real-world ingredient labels (image quality variations considered).

---

## Results and Performance

- OCR Accuracy: Approx. 90%+ for ingredient extraction from product images.
- Ingredient classification precision and recall rates >85%.
- Recommendation model accuracy around 95%.
- Detailed project report with evaluation metrics is available in the `docs/` folder.

---

## Acknowledgments

- Dr. M. Sailaja, Assistant Professor, Prasad V. Potluri Siddhartha Institute of Technology  
- Open-source libraries: EasyOCR, FuzzyWuzzy, scikit-learn, Streamlit  
- AICTE and Google for internship support  

---

## Contact

G. Jyothendra  
Email: gjyothendra@gmail.com  
LinkedIn: https://linkedin.com/in/jyothendraG  

---

🎉 Thank you for visiting the project repository! Feel free to raise issues or feature requests.
