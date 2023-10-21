f_host_door_selection_hyp = TabularCPD(    #A
    variable='Host Door Selection Hyp',    #A
    variable_card=3,    #A
    values=[    #A
        [0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,1,0],    #A
        [1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1],    #A
        [0,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0]    #A
    ],    #A
    evidence=['Coin Flip', 'Car Door Die Roll', '1st Choice Die Roll'],    #A
    evidence_card=[2, 3, 3],    #A
    state_names={    #A
        'Host Door Selection Hyp':['1st', '2nd', '3rd'],    #A
        'Coin Flip': ['tails', 'heads'],    #A
        'Car Door Die Roll': ['1st', '2nd', '3rd'],    #A
        '1st Choice Die Roll': ['1st', '2nd', '3rd']    #A
    }    #A
)    #A

f_strategy_hyp = TabularCPD(    #B
    variable='Strategy Hyp',    #B
    variable_card=2,    #B
    values=[[1, 0], [0, 1]],    #B
    evidence=['Coin Flip'],    #B
    evidence_card=[2],    #B
    state_names={    #B
        'Strategy Hyp': ['stay', 'switch'],    #B
        'Coin Flip': ['tails', 'heads']}    #B
)    #B

f_second_choice_hyp = TabularCPD(    #C
    variable='2nd Choice Hyp',    #C
    variable_card=3,    #C
    values=[    #C
        [1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0],    #C
        [0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1],    #C
        [0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0]    #C
    ],    #C
    evidence=['Strategy Hyp', 'Host Door Selection Hyp', '1st Choice Die Roll'],    #C
    evidence_card=[2, 3, 3],    #C
    state_names={    #C
        '2nd Choice Hyp': ['1st', '2nd', '3rd'],    #C
        'Strategy Hyp': ['stay', 'switch'],    #C
        'Host Door Selection Hyp': ['1st', '2nd', '3rd'],    #C
        '1st Choice Die Roll': ['1st', '2nd', '3rd']    #C
    }    #C
)    #C

f_win_or_lose_hyp = TabularCPD(    #D
    variable='Win or Lose Hyp',    #D
    variable_card=2,    #D
    values=[    #D
        [1,0,0,0,1,0,0,0,1],    #D
        [0,1,1,1,0,1,1,1,0],    #D
    ],    #D
    evidence=['2nd Choice Hyp', 'Car Door Die Roll'],    #D
    evidence_card=[3, 3],    #D
    state_names={    #D
        'Win or Lose Hyp': ['win', 'lose'],    #D
        '2nd Choice Hyp': ['1st', '2nd', '3rd'],    #D
        'Car Door Die Roll': ['1st', '2nd', '3rd']    #D
    }    #D
)    #D

#A Clone of the assignment function for the host door selection for use in the hypothetical world.
#B Clone of the assignment function for strategy for use in the hypothetical world.
#C Clone of the assignment function for the player's second choice for use in the hypothetical world.
#D Clone of the assignment function for the win/loss outcome of the game for use in the hypothetical world.
