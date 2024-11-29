# Senior Project

## Setup


Install all dependencies: 
```
pip install -r requirements.txt
```


Setup MySQL database instance: 
```
cd database_files
```


Create secret.py file with the accompanying info:
```
db_host = 'localhost'
db_user = 'root'
db_password = 'your_password_here'
db_database = 'Senior_Project'
```

```
python3 setup.py
```


To execute (Windows): 
```
python main.py
```


To execute (Mac/Linux): 
```
python3 main.py
```


To-Do:

    - Set MySQL password default to ON

    - Integrate Seb's page with Flask

    - Associate users with MySQL Reviews table
    
