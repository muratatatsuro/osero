#coding:utf-8


if __name__=='__main__':
    #学習回数
    n_epochs=1000
    #environment
    env=Osero()
    playerID=[env.black,env.white,env.black]

    #player agent
    players=[]
    #player[0]=env.black
    players.append(DQNagent(rows=env.n_rows,cols=env.n_cols))
    #player[1]=env.white
    players.append(DQNagent(rows=env.n_rows,cols=env.n_cols))
    #players[i]がdqnagentのインスタンスになる

    #ここから学習loop
    for epoch in range(n_epochs):
        #盤面初期化
        env.reset()
        #終了判定要素
        terminal=False
        
        #ゲームが終わるまでloop
        while terminal==False:
            for i in range(len(players)):
                #現在の盤面の状態
                state=env.cells
                targets=env.get_enables(playerID[i])
        
                #置く場所がある時
                if len(targets)>0:
                    #全ての行動(action)==trを試す
                    for tr in targets:
                        tmp=copy.deepcopy(env)
                        tmp.flip_able_man(tr,playerID[i])
                        tmp.flip()
                        win=tmp.winner()
                        end=tmp.End(playerID[i])
                
                        #次の状態に推移
                        state_x=tmp.cells
                        targets_x=tmp.get_enables(playerID[i+1])
                        if len(targets_x)==0:
                            targets_x=tmp.get_enables(playerID[i])
                            
                        #両者トレーニング
                        for j in range(len(players)):
                            reward=0
                            #ゲームでどちらかが買ったら
                            if end==True:
                                if win==playerID[j]:
                                    rewoad=1
                            
                            #ここまでの経験を後手のプレーヤーのものとして保存
                            players[j].store_experience(state,targets,tr,reward,state_x,targets_x,end)
                            #ミニバッチ学習
                            players[j].experience_replay()
                        #行動選択
                        action=players[i].epsilon_greedy(state,0.1,targets)
                        action=players[i].index2axis(action)
                        #行動を選択したら実行
                        env.flip_able_man(action,playerID[i])
                        env.flip()
            
                        #for log
                        loss=players[i].current_loss
                        Q_max,Q_action=players[i].next_able_action(state,targets)
            
                        print("player:{:1d} | pos:{} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(playerID[i], action, loss, Q_max))
            terminal=env.End(players[i])
        winner=env.winner()
        print("EPOCH: {:03d}/{:03d} | WIN: player{:1d}".format(e, n_epochs, w)
    players[1].save_model(n_epochs)
                
            
            

