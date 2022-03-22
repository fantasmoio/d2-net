#!/bin/sh
aws codeartifact login --tool pip --repository fantasmo --domain fantasmo --domain-owner 445621407199
aws codeartifact login --tool twine --repository fantasmo --domain fantasmo --domain-owner 445621407199
