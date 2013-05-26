#!/bin/bash
mvn compile

mvn dependency:build-classpath -Dmdep.outputFile=classpath.out

#for type in "left" "bottom"
for type in "left"
do
#	for train in "rand" "half"
for train in "half"
	do
		java -Xmx60g -cp ./target/classes:`cat classpath.out` edu.umd.cs.linqs.vision.OlivettiLatentTest $type $train| tee -a olivettiLatentLog.txt
	done
done