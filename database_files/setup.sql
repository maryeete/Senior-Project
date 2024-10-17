-- Drop the database if it exists and create a new one
DROP DATABASE IF EXISTS Senior_Project;
CREATE DATABASE Senior_Project;

-- Use the newly created database
USE Senior_Project;

-- Drop and create the User_Data table
DROP TABLE IF EXISTS User_Data;
CREATE TABLE User_Data (
    user_id          INT AUTO_INCREMENT PRIMARY KEY,  -- INT with auto-increment
    full_name        VARCHAR(150) NOT NULL,
    email            VARCHAR(150) UNIQUE NOT NULL,     -- Unique email
    username         VARCHAR(100) NOT NULL UNIQUE,     -- Unique username
    password         VARCHAR(205) NOT NULL,            -- Password
    date_created     TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL  -- Creation timestamp
);

-- Drop and create the Category table
DROP TABLE IF EXISTS Category;
CREATE TABLE Category (
    category_id      VARCHAR(150) UNIQUE PRIMARY KEY,
    category_name    VARCHAR(350)
);

-- Drop and create the Products table
DROP TABLE IF EXISTS Products;
CREATE TABLE Products (
    product_id       VARCHAR(100) NOT NULL PRIMARY KEY UNIQUE,
    product_name     VARCHAR(900) NOT NULL,
    category_id      VARCHAR(150),
    link             VARCHAR(599) NOT NULL,
    rating           FLOAT NOT NULL,
    total_reviews    INT NOT NULL,
    price            VARCHAR(100) NOT NULL,  
    
    CONSTRAINT products_fk_category FOREIGN KEY (category_id)
    REFERENCES Category(category_id)
);

-- Drop and create the Sentiments table
DROP TABLE IF EXISTS Sentiments;
CREATE TABLE Sentiments (
    sentiment_id     INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    sentiment        VARCHAR(300) NOT NULL 
);

-- Set the starting value for AUTO_INCREMENT in Sentiments
ALTER TABLE Sentiments AUTO_INCREMENT = 12050;

-- Drop and create the Reviews table
DROP TABLE IF EXISTS Reviews;
CREATE TABLE Reviews (
    user_id          INT NULL,  -- INT to match User_Data
    product_id       VARCHAR(100) NOT NULL,
    review_id        VARCHAR(100) NOT NULL PRIMARY KEY UNIQUE,
    rating           FLOAT NOT NULL,
    review_text      VARCHAR(3000),
    sentiment_id     INT,  
    
    CONSTRAINT reviews_fk_users FOREIGN KEY (user_id)
    REFERENCES User_Data(user_id),
   
    CONSTRAINT reviews_fk_products FOREIGN KEY (product_id)
    REFERENCES Products(product_id),
    
    CONSTRAINT reviews_fk_sentiments FOREIGN KEY (sentiment_id)
    REFERENCES Sentiments(sentiment_id)
);