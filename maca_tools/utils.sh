#!/bin/bash

function send_html_report()
{
    pip install junitparser
    xmls_dir=$(pwd)/build
    if [[ $# -gt 0 ]]; then
        xmls_dir=$1
    fi
    python $(dirname ${BASH_SOURCE[0]})/internal/sendmail.py $xmls_dir
}

function make_xmls_dir()
{
    if [ -d $1 ]; then
        rm -rf $1
    fi
    mkdir $1
}

put(){
    echo '<'${*}'>' >> $outfile
}

put_head(){
    put '?'${1}'?'
}

out_tabs(){
    tmp=0
    tabsstr=""
    while [ $tmp -lt $((tabs)) ]
    do
        tabsstr=${tabsstr}'\t'
        tmp=$((tmp+1))
    done
    echo -e -n $tabsstr >> $outfile
}

tag_start(){
    out_tabs
    put $1
    tabs=$((tabs+1))
}

tag_end(){
    tabs=$((tabs-1))
    out_tabs
    put '/'${1}
}

tag_value(){
    out_tabs
    str=""
    str=${1}' '''${2}'''/'
    put $str
}

generate_xml(){
    date=$(date +%Y%m%dT%H:%M:%S)
    outfile=$1
    tabs=0
    passed=$3
    failed=$4
    tests=$[ $passed + $failed ]
    splits=($(echo $2 | tr "#" "\n"))
    put_head 'xml version="1.0" encoding="UTF-8"'
    tag_start 'testsuites tests="'${tests}'" failures="'${failed}'" disabled="0" errors="0" time="0" timestamp="'${date}'" name="AllPyTests"'
    tag_start 'testsuite name="PyTest" tests="'${tests}'" failures="'${failed}'" disabled="0" skipped="0" errors="0" time="0" timestamp="'${date}'"'
    for split in "${splits[@]}"
    do
        values=($(echo ${split} | tr "," "\n"))
        case ${values[1]} in
            pass)
            tag_value 'testcase' 'name="'${values[0]}'" status="run" result="completed" time="'${values[2]}'" timestamp="'${date}'" classname="PyTest"';;
            fail)
            tag_start 'testcase name="'${values[0]}'" status="run" result="completed" time="'${values[2]}'" timestamp="'${date}'" classname="PyTest"'
            tag_start 'failure message="error!." type=""'
            tag_end 'failure'
            tag_end 'testcase'
            ;;
            *)
            ;;
        esac
    done
    tag_end 'testsuite'
    tag_end 'testsuites'
}
