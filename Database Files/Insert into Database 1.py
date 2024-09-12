import pandas as pd
import mysql.connector

# Connect to MySQL database
data = mysql.connector.connect(
    host='localhost',
    user='root',
    password='@MySeniorProJecT21',
    database='senior project'
)
cursor = data.cursor()

# Read data from CSV file
csv_file = "Final Amazon Products Data.csv"
df = pd.read_csv(csv_file)
print(df)

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
            data.commit()
            print("Review inserted successfully!")
        except mysql.connector.Error as err:
            print(f"Error inserting review: {err}")

# Close cursor and connection
cursor.close()
data.close()
