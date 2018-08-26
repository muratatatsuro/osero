
# coding: utf-8


import numpy as np
import random

class Osero():
    def __init__(self):
        self.n_rows=8
        self.n_cols=8
        self.cells=np.zeros((self.n_rows,self.n_cols))
        self.blank=0
        self.white=1
        self.black=2
        
    def reset(self):  
        self.cells[3][3]=self.white
        self.cells[3][4]=self.black
        self.cells[4][3]=self.black
        self.cells[4][4]=self.white
        
    def flip_able_man(self,now_state,current_player):
        self.now_state=now_state
        self.current_player=current_player
        dir_list=[-1,0,1]
        self.flip_able=[]
        for dx in dir_list:
            for dy in dir_list:
                temp=[]
                depth=0
                while True:
                    depth+=1
                    rx=self.now_state[0]+dx*depth
                    ry=self.now_state[1]+dy*depth
                    
                    if 0<=rx<8 and 0<=ry<8:
                        piece=self.cells[rx][ry]
                        
                        if piece==self.blank:
                            break
                        elif piece==self.current_player:
                            if temp!=[]:
                                self.flip_able.extend(temp)
                        else:
                            temp.append([rx,ry])
                    else:
                        break
        return self.flip_able
    
    def flip(self):
        for i in range(len(self.flip_able)):
            self.cells[self.flip_able[i][0]][self.flip_able[i][1]]=self.current_player
            
        self.cells[self.now_state[0]][self.now_state[1]]=self.current_player
        return self.cells
    
    def show_board(self):
        self.board=[]
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                pi=self.cells[i][j]
                if pi==self.blank:
                    self.board.append('*')
                elif pi==self.white:
                    self.board.append('○')
                else:
                    self.board.append('●')
                    
        self.board=np.array(self.board).reshape(self.n_rows,self.n_cols)
        print(self.board)
        
    def End(self,player):
        possible=[]
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                pi=self.cells[i][j]
                if pi!=self.blank:
                    if pi!=player:
                        possible.append([i,j])
        if len(possible)==0:
            return True
        else:
            return False
    def possible(self,player):
        self.possible=[]
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                pi=self.cells[i][j]
                if pi!=self.blank:
                    if pi!=player:
                        self.possible.append([i,j])
        return self.possible
        
    def player_change(self,player):
        if player==self.white:
            return self.black
        else:
            return self.white
        
    def winner(self):
        color_white=0
        color_black=0
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.cells[i][j]==self.white:
                    color_white+=1
                elif self.cells[i][j]==self.black:
                    color_black+=1
                else:
                    color_white+=0
                    color_black+=0
            
        if color_white>color_black:
            print('winner:white')
        elif color_white==color_black:
            print('on drow')
        else:
            print('winner:black')
            
    def show_index(self):
        def index_to_axis(index):
            r=int(index/8)
            c=index%8
            return [r,c]
        index_list=list(np.arange(0,self.n_rows*self.n_cols))
        axis_list=[]
        
        for i in range(0,self.n_rows*self.n_cols,self.n_rows):
            ind=index_list[i:i+self.n_rows]
            bb=[]
            for j in range(len(ind)):
                bb.append(index_to_axis(ind[j]))
            axis_list.append(bb)
            
        for i in range(self.n_rows):
            print(axis_list[i])
            
    def get_enables(self,color):
        """置ける位置のリストを返す関数"""
        result=[]
        enable_actions=np.arange(0,self.n_rows*self.n_cols)
        for action in enable_actions:
            r,c=self.index2axis(action)
            if self.cells[r][c]==self.blank:
                    result.append([r,c])
        return result
            
            
    #一次元のインデックスを二次元に変換        
    def index2axis(self,index):
        r=int(index/8)
        c=index%8
        return r,c
    #対戦者同士の得点を算出する関数
    def get_score(self,player):
        score=0
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                pi=self.cells[i][j]
                if pi==self.player:
                    score+=1
                else:
                    score+=0
        return score
