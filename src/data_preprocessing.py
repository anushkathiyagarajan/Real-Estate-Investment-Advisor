import pandas as pd
from sklearn.preprocessing import LabelEncoder

# LOAD DATA

df = pd.read_csv(
    r"C:\mini 2\data\raw\india_housing_prices.csv"
)

print("Original Shape:", df.shape)

# REMOVE DUPLICATES

df.drop_duplicates(inplace=True)

# HANDLE MISSING VALUES

df.ffill(inplace=True)

# FUTURE PRICE

growth_rate = 0.08

df['Future_Price_5Y'] = (

    df['Price_in_Lakhs']

    * ((1 + growth_rate) ** 5)

)

# APPRECIATION %

df['Appreciation_Percentage'] = (

    (

        df['Future_Price_5Y']

        - df['Price_in_Lakhs']

    )

    / df['Price_in_Lakhs']

) * 100

# GOOD INVESTMENT

median_price = df['Price_in_Lakhs'].median()

df['Good_Investment'] = (

    df['Price_in_Lakhs']

    < median_price

).astype(int)
print(df['Good_Investment'].value_counts())

# ENCODE CATEGORICAL COLUMNS

label_encoders = {}

categorical_cols = df.select_dtypes(
    include=['object']
).columns

for col in categorical_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

    label_encoders[col] = le

print("All categorical columns encoded successfully!")

# SAVE CLEANED DATA

df.to_csv(

    r"C:\mini 2\data\processed\cleaned_data.csv",

    index=False

)

print("Processed Shape:", df.shape)

print("✅ Cleaned data saved successfully!")