import csv
import sqlite3

class Sqldb:

    def __init__(self, train_data, ideal_data, recreated_test_data):
        self.train_data = train_data
        self.ideal_data = ideal_data
        self.recreated_test_data = recreated_test_data


    def train_sqlite_datebase(self):
        """
        Create exercise database in sqldb, and create a train table and insert data into.

        Args: 
            None

        Returns: 
            train data inserted into sqldb
        """

        try:

            with open(self.train_data, 'r') as fin:
                dr = csv.DictReader(fin)
                train_info = [(i['x'], i['y1'], i['y2'], i['y3'], i['y4']) for i in dr]
                print(train_info)

                # Connect to SQLite
                sqliteConnection = sqlite3.connect('excercise')
                cursor = sqliteConnection.cursor()

                # Create train table
                cursor.execute('create table if not exists train(x integer, y1 interger, y2 integer, y3 integer, y4 integer);')

                cursor.executemany(
                        "insert into train (x, y1, y2, y3, y4) VALUES (?, ?, ?, ?, ?);", train_info)

                # Show train table
                cursor.execute('select * from train')

                # View result
                result = cursor.fetchall()
                print(result)

                # Commit work and close connection
                sqliteConnection.commit()
                cursor.close()

        except sqlite3.Error as error:
            print('Error occurred - ', error)

        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print('SQLite Connection closed')



    def ideal_sqlite_database(self):
        """
        Create a ideal table in sqldb called exercise and insert data into.

        Args: 
            None

        Returns: 
            ideal data inserted into sqldb
        """
        try:

            with open(self.ideal_data, 'r') as fin:
                dr = csv.DictReader(fin)
                ideal_info = [(i['x'], i['y1'], i['y2'], i['y3'], i['y4'], i['y5'], i['y6'], i['y7'],
                                i['y8'], i['y9'], i['y10'], i['y11'], i['y12'], i['y13'], i['y14'],
                                i['y15'], i['y16'], i['y17'], i['y18'], i['y19'], i['y20'], i['y21'],
                                i['y22'], i['y23'], i['y24'], i['y25'], i['y26'], i['y27'], i['y28'],
                                i['y29'], i['y30'], i['y31'], i['y32'], i['y33'], i['y34'], i['y35'],
                                i['y36'], i['y37'], i['y38'], i['y39'], i['y40'], i['y41'], i['y42'],
                                i['y43'], i['y44'], i['y45'], i['y46'], i['y47'], i['y48'], i['y49'],
                                i['y50']) for i in dr]
                print(ideal_info)

                # Connect to SQLite
                sqliteConnection = sqlite3.connect('excercise')
                cursor = sqliteConnection.cursor()

                # Create train table
                cursor.execute('create table if not exists ideal(x integer, y1 interger, y2 integer, y3 integer, y4 integer, \
                                                                y5 integer, y6 interger, y7 integer, y8 integer, y9 integer, \
                                                                y10 integer, y11 interger, y12 integer, y13 integer, y14 integer, \
                                                                y15 integer, y16 interger, y17 integer, y18 integer, y19 integer, \
                                                                y20 integer, y21 interger, y22 integer, y23 integer, y24 integer, \
                                                                y25 integer, y26 interger, y27 integer, y28 integer, y29 integer, \
                                                                y30 integer, y31 interger, y32 integer, y33 integer, y34 integer, \
                                                                y35 integer, y36 interger, y37 integer, y38 integer, y39 integer, \
                                                                y40 integer, y41 interger, y42 integer, y43 integer, y44 integer, \
                                                                y45 integer, y46 interger, y47 integer, y48 integer, y49 integer, \
                                                                y50 integer);')

                cursor.executemany("insert into ideal (x, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47, y48, y49, y50) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", ideal_info)

                # Show train table
                cursor.execute('select * from ideal')

                # View result
                result = cursor.fetchall()
                print(result)

                # Commit work and close connection
                sqliteConnection.commit()
                cursor.close()

        except sqlite3.Error as error:
            print('Error occurred - ', error)

        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print('SQLite Connection closed')



    def test_sqlite_datebase(self):
        """
        Create a test table in sqldb called exercise and insert test data into.

        Args: 
            None

        Returns: 
            test data inserted into sqldb
        """
        try:

            with open(self.recreated_test_data, 'r') as fin:
                dr = csv.DictReader(fin)
                test_info = [(i['x'], i['y'], i['dev_y'], i['no_ideal_func']) for i in dr]
                print(test_info)

                # Connect to SQLite
                sqliteConnection = sqlite3.connect('excercise')
                cursor = sqliteConnection.cursor()

                # Create train table
                cursor.execute('create table if not exists test(x float, y float, dev_y float, no_ideal_func string);')

                cursor.executemany(
                        "insert into test (x, y, dev_y, no_ideal_func) VALUES (?, ?, ?, ?);", test_info)

                # Show train table
                cursor.execute('select * from test')

                # View result
                result = cursor.fetchall()
                print(result)

                # Commit work and close connection
                sqliteConnection.commit()
                cursor.close()

        except sqlite3.Error as error:
            print('Error occurred - ', error)

        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print('SQLite Connection closed')


if __name__ == "__main__":
    train_data = "data/train.csv"
    ideal_data = "data/ideal.csv"
    recreated_test_data = "data/recreted_test_data.csv"

    create_sqldb = Sqldb(train_data, ideal_data, recreated_test_data)

    create_sqldb.train_sqlite_datebase()
    create_sqldb.ideal_sqlite_database()
    create_sqldb.test_sqlite_datebase()