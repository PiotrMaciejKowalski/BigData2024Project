def initialize_spark():
    import os
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["SPARK_HOME"] = "/content/spark-3.5.0-bin-hadoop3"

    import findspark
    findspark.init()