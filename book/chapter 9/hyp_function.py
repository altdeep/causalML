from pgmpy.factors.discrete.CPD import TabularCPD


#A Clone of the assignment function for the host door selection for use in the hypothetical world.
f_host_door_selection_hyp = TabularCPD(
    variable='Host Door Selection Hyp',
    variable_card=3,
    values=[
        [0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,1,0],
        [1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1],
        [0,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0]
    ],
    evidence=['Coin Flip', 'Car Door Die Roll', '1st Choice Die Roll'],
    evidence_card=[2, 3, 3],
    state_names={
        'Host Door Selection Hyp':['1st', '2nd', '3rd'],
        'Coin Flip': ['tails', 'heads'],
        'Car Door Die Roll': ['1st', '2nd', '3rd'],
        '1st Choice Die Roll': ['1st', '2nd', '3rd']
    }
)

#B Clone of the assignment function for strategy for use in the hypothetical world.
f_strategy_hyp = TabularCPD(
    variable='Strategy Hyp',
    variable_card=2,
    values=[[1, 0], [0, 1]],
    evidence=['Coin Flip'],
    evidence_card=[2],
    state_names={
        'Strategy Hyp': ['stay', 'switch'],
        'Coin Flip': ['tails', 'heads']}
)

#C Clone of the assignment function for the player's second choice for use in the hypothetical world.
f_second_choice_hyp = TabularCPD(
    variable='2nd Choice Hyp',
    variable_card=3,
    values=[
        [1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0],
        [0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1],
        [0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0]
    ],
    evidence=['Strategy Hyp', 'Host Door Selection Hyp', '1st Choice Die Roll'],
    evidence_card=[2, 3, 3],
    state_names={
        '2nd Choice Hyp': ['1st', '2nd', '3rd'],
        'Strategy Hyp': ['stay', 'switch'],
        'Host Door Selection Hyp': ['1st', '2nd', '3rd'],
        '1st Choice Die Roll': ['1st', '2nd', '3rd']
    }
)

#D Clone of the assignment function for the win/loss outcome of the game for use in the hypothetical world.
f_win_or_lose_hyp = TabularCPD(
    variable='Win or Lose Hyp',
    variable_card=2,
    values=[
        [1,0,0,0,1,0,0,0,1],
        [0,1,1,1,0,1,1,1,0],
    ],
    evidence=['2nd Choice Hyp', 'Car Door Die Roll'],
    evidence_card=[3, 3],
    state_names={
        'Win or Lose Hyp': ['win', 'lose'],
        '2nd Choice Hyp': ['1st', '2nd', '3rd'],
        'Car Door Die Roll': ['1st', '2nd', '3rd']
    }
)
