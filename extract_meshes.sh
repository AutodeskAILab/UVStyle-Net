#!/bin/zsh

INPUT_DIR=~/brep_style/solidmnist/mesh/test
TEMP=~/meshes_temp
TARGET_DIR=~/brep_style/solidmnist/mesh/test_extracted

echo Extracting meshes...
mkdir -p $TEMP
for FILE in $INPUT_DIR/**.zip ; do
    unzip "$FILE" -d $TEMP
done

echo Copying meshes...
mkdir -p $TARGET_DIR
for FILE in $TEMP/**/*.stl ; do
    cp "$FILE" $TARGET_DIR/"$(basename $FILE)"
done

echo Cleaning up...
rm -rf $TEMP


