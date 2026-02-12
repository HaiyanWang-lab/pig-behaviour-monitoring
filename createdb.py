import pymysql

class PigDB():
    def __init__(self):
        self.conn = pymysql.connect(host='localhost', user='root', password='123456', charset='utf8mb4')
        self.cursor = self.conn.cursor()
    def createDb(self):
        sql = "CREATE DATABASE IF NOT EXISTS pig_3"
        self.cursor.execute(sql)
        self.cursor.execute("use pig_3")


    # stand  站；  side 侧躺；    prone 躺
    def createTable(self):
        sql_2 = '''CREATE TABLE `piginfo` (
                 `id` INT AUTO_INCREMENT,
                 `videoid` TEXT(100) ,  
                  `pigid` TEXT(100) ,  
                 `time` TEXT(200) ,   
                 `standing` TEXT(2000) ,   
                 `sidelying` TEXT(2000) , 
                 `pronelying` TEXT(2000) , 
                 `videoname` TEXT(100) ,  
                 PRIMARY KEY (`id`)
               ) ENGINE=InnoDB;
               '''
        try:
            self.cursor.execute(sql_2)
            print("创建数据库成功")
        except Exception as e:
            print("创建数据库结果：case%s" % e)
        finally:
            pass
            # cursor.close()
            # db.close()
    def insertInto(self,videoid,pigid, time,stand,side,prone,videoname):
        sql = "INSERT INTO  piginfo VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
        params = [
            (0,videoid, pigid,time,stand,side,prone,videoname), ]
        try:
            self.cursor.executemany(sql,params)
            self.conn.commit()
            print("有",self.cursor.rowcount,"插入数据成功")
        except Exception as e:
            print("插入数据失败：case%s"%e)
            self.conn.rollback()
        finally:
            pass
            # cursor.close()
            # db.close()
    def selectall(self):
        sql = "SELECT * FROM piginfo order by id DESC"
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            return results
        except Exception as e:
            print("查询出错：case%s"%e)
            return None

    def selecByPigId(self,pigid):
        sql = "SELECT * FROM piginfo where pigid = '%s' "%(pigid)
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            return results
        except Exception as e:
            print("查询出错：case%s" % e)
            return None

    def selecByIds(self,pigid,videoname):
        sql = "SELECT * FROM piginfo where pigid = '%s' and videoname = '%s'"%(pigid,videoname)
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            return results
        except Exception as e:
            print("查询出错：case%s" % e)
            return None

    def selecByVideoId(self,videoname):
        sql = "SELECT * FROM piginfo where videoname = '%s' "%(videoname)
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            return results
        except Exception as e:
            print("查询出错：case%s" % e)
            return None

    def clearDb(self,pigid= None):
        if pigid is None:
            sql = """DELETE FROM piginfo"""
        else:
            sql = """DELETE FROM piginfo WHERE pigid = %s""" % (pigid)
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            print("删除数据成功")

        except Exception as e:
            print("删除数据失败：case%s" % e)
            self.conn.rollback()

if __name__ == "__main__":
    try:
        pigDb = PigDB()
        pigDb.createDb()
        pigDb.createTable()
        #faceDb.insertInto("test","info","0,0")
        pigDb.clearDb(None)
        result = pigDb.selectall( )
        if result is not None:
            for row in result:
                print(row)
                # id = row[0]
                # name = row[1]
                # vector = row[3]
                # print("id=%s,name=%s,vector=%s" % \
                #       (id, name, vector))
    except Exception as e:
        pass
    finally:
        pigDb.cursor.close()
        pigDb.conn.close()
