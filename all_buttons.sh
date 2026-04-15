!/bin/bash                                                                                                                                                                

mapfile BUTTONS <<-EOF     
1 MPC Procedure
2 get drone position
3 get recorded trajectory
0 Exit                                                                                                                                                                 
EOF

for BUTTON in "${BUTTONS[@]}"
do
    VALUE=("${BUTTON[@]%% *}")
    TEXT=("${BUTTON[@]#* }")
    echo VALUE=${VALUE}
    echo TEXT=${TEXT}

    convert -size 160x200 -background lightblue -pointsize 30 -fill black -gravity Center caption:"$TEXT" -flatten /tmp/button_${VALUE}.png

    YADPARAMS+="--button=!/tmp/button_${VALUE}.png:$VALUE "

done

while true
do

    yad --form ${YADPARAMS[@]}
    CODE=$?

    if [[ $CODE == 0 ]]
    then
        break
    fi


    rostopic pub -1 /ui_command std_msgs/Int16 $CODE

done

