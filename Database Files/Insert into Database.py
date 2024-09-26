import pandas as pd
import mysql.connector
from mysql.connector import IntegrityError
from secret import host, user, password, database

# Connect to MySQL database
data = mysql.connector.connect(
    host=host,
    user=user,
    password=password,
    database=database
)

cursor = data.cursor()
csv_file = "Final Amazon Products Data.csv"
df = pd.read_csv(csv_file)
print(df)

# Insert data into Category table
for _, row in df[['Category ID', 'Category Name']].drop_duplicates().iterrows():
    try:
        cursor.execute("INSERT INTO Category (category_id, category_name) VALUES (%s, %s)", tuple(row))
    except IntegrityError:
        pass  # Ignore the duplicate key error and continue inserting

# Insert data into Products table
for _, row in df[['Product ID', 'Product Name', 'Category ID', 'Link', 'Rating', 'Total Reviews', 'Price']].drop_duplicates().iterrows():
    cursor.execute("INSERT INTO Products (product_id, product_name, category_id, link, rating, total_reviews, price) VALUES (%s, %s, %s, %s, %s, %s, %s)", tuple(row))


# Commit changes and close cursor
data.commit()
cursor.close()

# Close connection
data.close()