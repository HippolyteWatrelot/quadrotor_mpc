!/bin/bash

mapfile BUTTONS <<-EOF                                                                                                                                     
1 start record
2 stop record
3 plot path
0 Exit                                                                                                                                                                 
EOF

for BUTTON in "${BUTTONS[@]}"
do
    VALUE=("${BUTTON[@]%% *}")
    TEXT=("${BUTTON[@]#* }")
    echo VALUE=${VALUE}
    echo TEXT=${TEXT}

    convert -size 80x200 -background lightblue -pointsize 10 -fill black -gravity Center caption:"$TEXT" -flatten /tmp/button_${VALUE}.png

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


    if [[ $CODE == 3 ]]
    then
        rosrun quadrotor_mpc plot_robot_path.py
    fi



    rostopic pub -1 /command std_msgs/Int16 $CODE

done

