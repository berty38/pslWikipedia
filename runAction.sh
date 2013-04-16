#!/bin/bash
mvn compile

mvn dependency:build-classpath -Dmdep.outputFile=classpath.out

java -Xmx8g -cp ./target/classes:`cat classpath.out` edu.umd.cs.linqs.action.Action1_DataLoader
for ((i = 0 ; i < 4; i++))
do
	java -Xmx32g -cp ./target/classes:`cat classpath.out` edu.umd.cs.linqs.action.ActionRecog1 $i
done

