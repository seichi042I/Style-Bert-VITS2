#!/bin/bash

# Data_storeディレクトリ内のサブディレクトリのリストを取得して変数に代入
spkdirs=$(find Data_store -mindepth 1 -maxdepth 1 -type d)
mkdir -p Data/$1/raw
# サブディレクトリの数だけループを回す
for dir in $spkdirs; do
    # サブディレクトリ内に"transcript_utf8.txt"が存在するか確認
    if [ -f "$dir/transcript_utf8.txt" ]; then
        # ファイルが存在する場合は内容を処理する
        echo $dir
        count=0
        # デリミタを":"に設定して、ファイルの内容をwhileループで処理する
        while IFS=":" read -r file_name content; do
            # file_nameとcontentを使って何か処理を行う（ここでは例として表示する）

            # 音声ファイルの長さを取得し、1.0以上であれば1を返す条件式の結果をenough_lengthに格納する
            if [ ! -e "Data_store/${dir##*/}/wav/${file_name}.wav" ];then
                continue
            fi
            duration=$(soxi -D "Data_store/${dir##*/}/wav/${file_name}.wav")
            enough_length=$(echo "$duration > 0.25" | bc)
            # contentの文字数が3以上128未満でかつenough_lengthが1である場合に特定の処理をする
            if [ ${#content} -gt 1 ] && [ ${#content} -lt 128 ] && [ $enough_length -eq 1 ]; then
                if [[ ! "$変数" =~ [[:space:]] ]];then
                    count=$((count + 1))
                    echo -en "${count}\r"
                    echo "${dir##*/}_${file_name}.wav|${dir##*/}|JP|${content}" >> Data/$1/esd.list;
                    cp ${dir}/wav/${file_name}.wav Data/$1/raw/${dir##*/}_${file_name}.wav
                else
                    echo "$content"
                fi
            else
                echo "condition not satisfied: ${file_name}, ${content}"
            fi
        done < "$dir/transcript_utf8.txt"
        echo ""
    fi
done