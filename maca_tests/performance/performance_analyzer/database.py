from playhouse.mysql_ext import MySQLConnectorDatabase
from peewee import CharField
from peewee import DateTimeField
from peewee import TimeField
from peewee import IntegerField
from peewee import BigIntegerField
from peewee import DoubleField
from peewee import Model

db = MySQLConnectorDatabase(None)

def dataBase():
    return db

class BaseModel(Model):
    class Meta:
        database = db

class Report(Model):
    date = DateTimeField()
    function = CharField()
    testcase = CharField()
    status = CharField()
    teststart = DateTimeField()
    testend = DateTimeField()
    duration = TimeField()
    testgroup = CharField()
    feature = CharField()
    hardware = CharField()
    job_name = CharField()
    software_version = CharField()
    branch = CharField()
    build_url = CharField()
    real_time = DoubleField()
    cpu_time = DoubleField()
    threads = IntegerField()
    metric1 = DoubleField()
    metric2 = DoubleField()
    metric3 = DoubleField()
    metric4 = DoubleField()
    metric5 = DoubleField()
    golden1 = DoubleField()
    golden2 = DoubleField()
    golden3 = DoubleField()
    golden4 = DoubleField()
    golden5 = DoubleField()
    arg1 = BigIntegerField()
    arg2 = BigIntegerField()
    arg3 = BigIntegerField()
    arg4 = BigIntegerField()
    arg5 = BigIntegerField()
    arg6 = BigIntegerField()
    arg7 = BigIntegerField()
    arg8 = BigIntegerField()
    arg9 = BigIntegerField()
    arg10 = BigIntegerField()

    class Meta:
        database = db
        table_name = "reports"


class Target(Model):
    job_name = CharField()
    testcase = CharField()
    metric1_name = CharField()
    metric2_name = CharField()
    metric3_name = CharField()
    metric4_name = CharField()
    metric5_name = CharField()

    class Meta:
        database = db
        table_name = "targets"