#!/bin/bash
mvn compile

mvn dependency:build-classpath -Dmdep.outputFile=classpath.out

java -Xmx8g -cp ./target/classes:`cat classpath.out` edu.umd.cs.linqs.action.Action1_DataLoader
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.umd.cs.linqs.action.ActionRecog1
#for ((i = 0 ; i < 1; i++))
#do
#	java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.umd.cs.linqs.action.ActionRecog1 $i
#done

