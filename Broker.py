import streamlit as st
import joblib

# Load the trained model
model = joblib.load('models/apps/Broker/xgboost_model.pkl')

def predict_price(features):
    feature_list = [features[feature] for feature in ['number of bedrooms', 'number of bathrooms', 'number of floors', 'number of views',
                                                      'Area of the house(excluding basement)', 'Area of the basement', 'Built Year', 'Renovation Year',
                                                      'Number of schools nearby', 'Distance from the airport', 'average condition grade', 'total area']]
    predicted_price = model.predict([feature_list])[0]
    return predicted_price

def main():
    st.title('House Price Prediction')

    # User input for each feature
    number_of_bedrooms = st.number_input('Number of bedrooms', min_value=1.0, step=1.0)
    number_of_bathrooms = st.number_input('Number of bathrooms', min_value=1.0, step=0.5)
    number_of_floors = st.number_input('Number of floors', min_value=1.0, step=1.0)
    number_of_views = st.number_input('Number of views', min_value=0, step=1)
    area_house_excluding_basement = st.number_input('Area of the house (excluding basement)', min_value=0.0)
    area_of_basement = st.number_input('Area of the basement', min_value=0.0)
    built_year = st.number_input('Built Year', min_value=0)
    renovation_year = st.number_input('Renovation Year', min_value=0)
    number_of_schools_nearby = st.number_input('Number of schools nearby', min_value=0, step=1)
    distance_from_the_airport = st.number_input('Distance from the airport', min_value=0.0)
    average_condition_grade = st.number_input('Average condition grade', min_value=1.0, max_value=10.0, step=0.5)
    total_area = st.number_input('Total area', min_value=0.0)

    # Make prediction when button is clicked
    if st.button('Predict'):
        features = {
            'number of bedrooms': number_of_bedrooms,
            'number of bathrooms': number_of_bathrooms,
            'number of floors': number_of_floors,
            'number of views': number_of_views,
            'Area of the house(excluding basement)': area_house_excluding_basement,
            'Area of the basement': area_of_basement,
            'Built Year': built_year,
            'Renovation Year': renovation_year,
            'Number of schools nearby': number_of_schools_nearby,
            'Distance from the airport': distance_from_the_airport,
            'average condition grade': average_condition_grade,
            'total area': total_area
        }
        predicted_price = predict_price(features)
        st.write(f'Predicted price of the house: Rs. {predicted_price:.2f}')

if __name__ == '__main__':
    main()
