#!/bin/bash
mvn compile

mvn dependency:build-classpath -Dmdep.outputFile=classpath.out

for type in "left" "bottom"
do
	for train in "rand"
	do
		java -Xmx30g -cp ./target/classes:`cat classpath.out` edu.umd.cs.linqs.vision.CaltechTest $type $train| tee -a caltechLog.txt
	done
done