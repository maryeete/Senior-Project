import pandas as pd
import mysql.connector
from mysql.connector import IntegrityError
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from secret import db_host, db_user, db_password, db_database


def execute_sql(cursor, statement):
    try:
        cursor.execute(statement)

        # If the command is a SELECT statement, fetch the results
        if statement.lower().startswith('select'):
            cursor.fetchall()  # Fetch all results to avoid unread results
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        print(f"SQL statement that caused the error: {statement}")


"""
SECTION: Initial Database setup
"""
# Connect to the MySQL server (without specifying a database initially)
data = mysql.connector.connect(
    host=db_host,
    user=db_user,
    password=db_password,
)

cursor = data.cursor()

with open('setup.sql', 'r') as sql_file:
    sql_script = sql_file.read()

# Split the script by semicolons and execute each statement individually
sql_commands = sql_script.split(';')

for statement in sql_commands:
    statement = statement.strip()
    if statement:
        execute_sql(cursor, statement)

try:
    data.commit()
except mysql.connector.Error as err:
    print(f"Error during commit: {err}")


"""
SECTION: Insert into products table
"""
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

try:
    data.commit()
except mysql.connector.Error as err:
    print(f"Error during commit: {err}")


"""
SECTION: Insert into reviews table
"""
# Iterate over each row in the DataFrame
for _, row in df.iterrows():
    product_id = row['Product ID']
    reviews = row['Reviews']
    review_ids = row['ReviewID']

    # Split reviews and review IDs
    review_texts = reviews.split('Rating: ')[1:]
    review_ids_list = review_ids.split(', ')

    # Ensure the number of reviews matches the number of review IDs
    if len(review_texts) != len(review_ids_list):
        print(f"Error: Number of reviews does not match number of review IDs for product {product_id}")
        continue

    # Insert each review into the database
    for review_text, review_id in zip(review_texts, review_ids_list):
        rating, review = review_text.split(',', 1)
        rating = float(rating.strip())  # Convert rating to float

        # Extract the review text part
        review = review.split('Review: ')[1].strip()

        try:
            # Insert review into the database
            insert_query = "INSERT INTO Reviews (product_id, rating, review_text, review_id) VALUES (%s, %s, %s, %s)"
            cursor.execute(insert_query, (product_id, rating, review, review_id.strip()))
            print("Review inserted successfully!")  # Debug output
        except mysql.connector.Error as err:
            print(f"Error inserting review: {err}")

# Commit all changes to the database
try:
    data.commit()
except mysql.connector.Error as err:
    print(f"Error during commit: {err}")

# Close cursor and connection
cursor.close()
data.close()
