import sqlite3 as sl
# sqlite using for create database.


class Db_Operations:
    # This class using for arrange database operationas.

    def __init__(self, user_name, db_type, cl1, cl2, cl3):
        """Create db and add a table in it. Takes 3 paramether:
           1. user_name: Create dbwith user name. 
           2. db_type: Determine db_type to use db what for. 
           3. cl1 cl2 cl3: Takes column name from user .
        """

        user_name, db_type = Admin.arrange_string(
            user_name), Admin.arrange_string(db_type)
        con = sl.connect("{}.db".format(user_name))
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS '{}' (id, {}, {}, {})".format(db_type, cl1,
                                                                      cl2, cl3))
        con.commit()
        con.close()

    @staticmethod
    def read_data(user_name, db_type):
        # To read data in table and return this data.

        con = sl.connect("{}.db".format(user_name))
        cur = con.cursor()
        cur.execute("SELECT * FROM '{}'".format(db_type))
        read = cur.fetchall()
        con.commit()
        con.close()
        return read

    @staticmethod
    def add_data(user_name, db_type, vl1, vl2, vl3):
        # Adding data in table of database.
        # vl1, vl2, vl3: Takes values to add data in table from user.

        con = sl.connect("{}.db".format(user_name))
        cur = con.cursor()
        data_id = Db_Operations.read_data(user_name, db_type)
        if data_id:
            cur.execute("INSERT INTO '{}' VALUES (?,?,?,?)".format(db_type), (len[data_id]+1, vl1,
                                                                              vl2, vl3))
        else:
            cur.execute("INSERT INTO '{}' VALUES (?,?,?,?)".format(db_type), (1, vl1,
                                                                              vl2, vl3))
        con.commit()
        con.close()

class Admin:
    # This class using for admin operataion like control user inforamtion, register user,
    # and arrange input strins.

    def admin_base():
        con = sl.connect("admin_base.db")
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS 'Users Data' (id, name, s_name, user_name, password)")

    def insert_user(name, s_name, user_name, password):
        # User gives persnal information to register and this function insert data into table.

        user_infos = [name, s_name, user_name, password]
        for i in range(len(user_infos)):
            user_infos[i] = Admin.arrange_string(user_infos[i])

        con = sl.connect("admin_base.db")
        cur = con.cursor()
        cur.execute("SELECT * FROM 'Users Data' WHERE id")
        read = cur.fetchall()
        cur.execute("INSERT INTO 'Users Data' Values (?,?,?,?,?)",
                    (len(read)+1, name, s_name, user_name, password))
        con.commit()
        con.close()

    def ctrl_user(user_name, password):
        # Controls given log in information from user. 
        
        user_data = Db_Operations.read_data("admin_base", "Users Data")
        for i in range(len(user_data)):
            if user_data[i][3] == user_name and user_data[i][4] == password:
                flag = 1
                return flag

    def arrange_string(stringg):
        # Arranges strgins in program. 
        
        stringg = stringg.replace(" ", "_").lower()
        return stringg
    


Admin.admin_base()
# Creats database for users' data. If no exists. 

print("""
=========================
WELCOME TO YOUR DATABASES
=========================

ATTENTION: If you do nat have an account, you must create one first.

Log in => 1
Register => 2
Exite => 0
      """)
user = input("Command Number: ")
if user == "1":
    user_name = input("User Name: ")
    password = input("Password: ")

    if Admin.ctrl_user(user_name, password):
        print("\nWelcome user {}".format(user_name))
        while True:
            print("\nCreate Database => 1\nShow Database => 2\nAdd Data => 3\nExit => 0")
            db_loc = input("Command Number: ")
            if db_loc == "1":
                print("Attantion: You have 3 columns to name\n\n")
                db_type = input("DB Type: ")
                cl1 = input("1. Column Name: ")
                cl2 = input("2. Column Name: ")
                cl3 = input("3. Column Name: ")
                Db_Operations(user_name, db_type, cl1, cl2, cl3)
                print("Database created.")
            elif db_loc == "2":
                db_type = input("DB Type: ")
                print("\nYour Data of {} Database".format(db_type))
                data = Db_Operations.read_data(user_name, db_type)
                for i in range(len(data)):
                    print(data[i][0], data[i][1], data[i][2], data[i][3])
            elif db_loc == "3":
                db_type = input("DB Type: ")
                val1 = input("1. Column's Values: ")
                val2 = input("2. Column's Value: ")
                val3 = input("3. Column's Value: ")
                Db_Operations.add_data(user_name, db_type, val1, val2, val3)
                print("Inserting values...")
                print("Values added to your database.")
            elif db_loc == "0":
                print("Exiting...")
                break
            else:
                print("Invalid Command !!!")

    else:
        print("Wrong user name or password!!!")


elif user == "2":
    name = input("Name: ")
    s_name = input("Surname: ")
    user_name = input("User Name: ")
    password = input("Password: ")
    Admin.insert_user(name, s_name, user_name, password)
    print("\nRegistiration Successful.\n\nRun the program again and log in to use databases")
elif user == "0":
    print("Exiting...")
else:
    print("Invalid Command!!!")
