import sqlite3

# connect to sqlite3 database
connection = sqlite3.connect("student.db")

# create a cursor object to create a table, insert record
cursor = connection.cursor()

# create a table
table_info = """
create table student(Name varchar(25), Class varchar(25), Section varchar(25), Marks int)
"""

cursor.execute(table_info)


# inserting some records to table
cursor.execute("Insert into student values ('krish','Data Science', 'A',90)")
cursor.execute("Insert into student values ('Dev','Data Science', 'B',100)")
cursor.execute("Insert into student values ('Sawab','Data Science', 'A',86)")
cursor.execute("Insert into student values ('Dhanesh','Devops', 'A',94)")
cursor.execute("Insert into student values ('Pratik','Devops', 'B',67)")


# display all the records
print("The inserted records are")
data = cursor.execute("select * from student")
for row in data:
    print(row)


# commit your changes in the databases
connection.commit()
connection.close()