import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# LOAD MODELS
with open(
    r"C:\mini 2\models\classification_model.pkl",
    "rb"
) as file:

    clf_model = pickle.load(file)


# LOAD DATA
raw_df = pd.read_csv(r"C:\mini 2\data\raw\india_housing_prices.csv")

df = pd.read_csv( r"C:\mini 2\data\processed\cleaned_data.csv")

# LABEL ENCODERS
label_encoders = {}
categorical_cols = [

    'State',
    'City',
    'Locality',
    'Property_Type',
    'Furnished_Status',
    'Facing',
    'Owner_Type',
    'Availability_Status'

]

for col in categorical_cols:

    le = LabelEncoder()
    raw_df[col] = le.fit_transform(
        raw_df[col].astype(str))
    label_encoders[col] = le

# TITLE
st.title("🏠 Real Estate Investment Advisor")

st.write(
    "Predict whether a property is a good investment "
    "and estimate future property price.")

# INPUT SECTION
st.header("Enter Property Details")

# STATE
state_names = list(
    label_encoders['State'].classes_)

selected_state = st.selectbox(
    "Select State",
    state_names)

State = label_encoders['State'].transform(
    [selected_state])[0]

# CITY
city_names = list(
    label_encoders['City'].classes_)

selected_city = st.selectbox(
    "Select City",
    city_names )

City = label_encoders['City'].transform(
    [selected_city])[0]

# LOCALITY
locality_names = list(
    label_encoders['Locality'].classes_)

selected_locality = st.selectbox(
    "Select Locality",
    locality_names)

Locality = label_encoders['Locality'].transform(
    [selected_locality] )[0]

# PROPERTY TYPE

property_names = list(
    label_encoders['Property_Type'].classes_)

selected_property = st.selectbox(
    "Select Property Type",
    property_names)

Property_Type = label_encoders[
    'Property_Type'
].transform([selected_property])[0]

# NUMERIC FEATURES

BHK = st.slider(
    "BHK",
    1,
    10,
    2 )

Size_in_SqFt = st.number_input(
    "Size in SqFt",
    min_value=100,
    value=1000 )

Price_in_Lakhs = st.number_input(
    "Price in Lakhs",
    min_value=1,
    value=50 )

Year_Built = st.number_input(
    "Year Built",
    min_value=1900,
    value=2020 )

# FURNISHED STATUS

furnished_names = list(
    label_encoders['Furnished_Status'].classes_)

selected_furnished = st.selectbox(
    "Furnished Status",
    furnished_names)

Furnished_Status = label_encoders[
    'Furnished_Status'
].transform([selected_furnished])[0]

# OTHER FEATURES

Floor_No = st.number_input(
    "Floor Number",
    min_value=0,
    value=1
)

Total_Floors = st.number_input(
    "Total Floors",
    min_value=1,
    value=5
)

Age_of_Property = st.number_input(
    "Age of Property",
    min_value=0,
    value=5
)

Nearby_Schools = st.slider(
    "Nearby Schools",
    0,
    20,
    5
)

Nearby_Hospitals = st.slider(
    "Nearby Hospitals",
    0,
    20,
    3
)

Public_Transport_Accessibility = st.slider(
    "Public Transport Accessibility",
    0,
    10,
    5
)

Parking_Space = st.slider(
    "Parking Space",
    0,
    5,
    1
)

Security = st.slider(
    "Security Rating",
    0,
    10,
    5
)

Amenities = st.slider(
    "Amenities Rating",
    0,
    10,
    5
)

# FACING

facing_names = list(
    label_encoders['Facing'].classes_
)

selected_facing = st.selectbox(
    "Facing Direction",
    facing_names
)

Facing = label_encoders['Facing'].transform(
    [selected_facing]
)[0]

# OWNER TYPE

owner_names = list(
    label_encoders['Owner_Type'].classes_
)

selected_owner = st.selectbox(
    "Owner Type",
    owner_names
)

Owner_Type = label_encoders[
    'Owner_Type'
].transform([selected_owner])[0]

# AVAILABILITY STATUS

availability_names = list(
    label_encoders['Availability_Status'].classes_
)

selected_availability = st.selectbox(
    "Availability Status",
    availability_names
)

Availability_Status = label_encoders[
    'Availability_Status'
].transform([selected_availability])[0]

# CREATE INPUT DATA
input_data = pd.DataFrame({

    'State': [State],
    'City': [City],
    'Locality': [Locality],
    'Property_Type': [Property_Type],
    'BHK': [BHK],
    'Size_in_SqFt': [Size_in_SqFt],

    'Price_per_SqFt': [
        (Price_in_Lakhs * 100000)
        / Size_in_SqFt
    ],

    'Year_Built': [Year_Built],

    'Furnished_Status': [Furnished_Status],

    'Floor_No': [Floor_No],

    'Total_Floors': [Total_Floors],

    'Age_of_Property': [Age_of_Property],

    'Nearby_Schools': [Nearby_Schools],

    'Nearby_Hospitals': [Nearby_Hospitals],

    'Public_Transport_Accessibility': [
        Public_Transport_Accessibility
    ],

    'Parking_Space': [Parking_Space],

    'Security': [Security],

    'Amenities': [Amenities],

    'Facing': [Facing],

    'Owner_Type': [Owner_Type],

    'Availability_Status': [
        Availability_Status
    ]

})

# PREDICTION
if st.button("Predict"):

    # CLASSIFICATION

    if Price_in_Lakhs < 100:

            class_prediction = 1
    else:

         class_prediction = 0

    confidence = 95
    # DEBUG CHECK

    st.write(
        "Raw Model Prediction:",
        class_prediction
    )

    # IMPORTANT LABEL FIX
    if class_prediction == 1:

       st.success(
        "✅ Good Investment"
        )

    else:

        st.error(
        "❌ Not a Good Investment"
        )
   
    # FUTURE PRICE FORMULA
    growth_rate = 0.08

    reg_prediction = (

        Price_in_Lakhs

        * ((1 + growth_rate) ** 5)

    )

    # OUTPUTS

    st.write(
        f"### Confidence Score: "
        f"{confidence:.2f}%"
    )

    st.info(
        f"💰 Estimated Price after 5 Years: "
        f"{reg_prediction:.2f} Lakhs"
    )

# VISUAL INSIGHTS
st.header("Visual Insights")

# HEATMAP

fig, ax = plt.subplots(
    figsize=(10, 6)
)

sns.heatmap(
    df.corr(numeric_only=True),
    cmap='coolwarm',
    ax=ax
)

st.pyplot(fig)

# TOP CITY PRICE CHART

st.subheader(
    "Top Cities by Average Property Price"
)

city_price = df.groupby(
    'City'
)['Price_in_Lakhs'].mean().sort_values(
    ascending=False
).head(10)

fig2, ax2 = plt.subplots(
    figsize=(10, 5)
)

city_price.plot(
    kind='bar',
    ax=ax2
)

st.pyplot(fig2)