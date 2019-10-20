commit_statement="add "
commit_statement=$commit_statement$1


git add *
git commit -m "$commit_statement"
git push https://yuxuan:yuxuan123@git.ram-lab.com/yuxuan/My_papers.git