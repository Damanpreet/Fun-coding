#!/bin/bash
INPUT_FILE=users_info.csv
OLDIFS=$IFS
IFS=','

[ ! -f $INPUT_FILE ] && { echo "$INPUT_FILE file not found."; exit 99; }

while read userid branch remote_repo
do
    echo "userid:  $userid"
    echo "branch: $branch"
    echo "remote repository: $remote_repo"
    if [[ ! -d $userid ]]
    then
        mkdir ./$userid
        echo "$userid directory created"
    fi
    cd $userid
    git clone --branch $userid-$branch $remote_repo || error=true

    if [ $error ]
    then
        echo "check if branch $branch exists for the user $userid. Repository link: $remote_repo"
        #exit -1
    fi
    cd ..
done < $INPUT_FILE
