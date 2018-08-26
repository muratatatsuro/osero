
# coding: utf-8

#強化学習 脳みそマン作成
import os
import numpy as np
import tensorflow as tf
from collections import deque

class DQNagent(object):
    """クラスラベルはself.n_actions"""
    def __init__(self,rows,cols):
        self.enable_actions=np.arange(0,64).tolist()
        self.hidden_layer_size=30
        self.n_actions=len(self.enable_actions)
        self.rows=rows
        self.cols=cols
        
        #minibatch learning
        self.batch_size=32
        #学習回数
        self.replay_memory_size=1000
        #学習率
        self.learning_rate=0.01
        #割引率
        self.discount_factor=0.9
        #exploration
        self.exploration=0.1
        
        #遷移D
        self.D=deque(maxlen=self.replay_memory_size)
        
        #モデルの初期化
        self.init_model()
        #損失関数
        self.current_loss=0.0
        
    #隠れ層1の多層パーセプトロン    
    def init_model(self):
        #二次元
        self.x=tf.placeholder(tf.float32,shape=[None,self.rows,self.cols])
        #教師ラベル
        self.y_=tf.placeholder(tf.float32,[None,self.n_actions])
        #ニューラルネットワークを実装する際には一次元変換
        x_flat=tf.reshape(self.x,[-1,self.rows*self.cols])
        
        #全結合層
        size=self.rows*self.cols
        #第一層
        w_fc=tf.Variable(tf.truncated_normal([size,size],stddev=0.01))
        b_fc=tf.Variable(tf.zeros([size]))
        h_fc=tf.nn.relu(tf.matmul(x_flat,w_fc)+b_fc)
        
        #第二層
        w_fc2=tf.Variable(tf.truncated_normal([size,self.hidden_layer_size],stddev=0.01))
        b_fc2=tf.Variable(tf.zeros([self.hidden_layer_size]))
        h_fc2=tf.nn.relu(tf.matmul(h_fc,w_fc2)+b_fc2)
        
        #出力層
        w_out=tf.Variable(tf.truncated_normal([self.hidden_layer_size,self.n_actions],stddev=0.01))
        b_out=tf.Variable(tf.zeros([self.n_actions]))
        self.y=tf.matmul(h_fc2,w_out)+b_out
        
        #損失関数
        self.loss=tf.reduce_mean(tf.square(self.y_-self.y))
        
        #最適化指標
        optimizer=tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer=optimizer.minimize(self.loss)
        
        #保存インスタンス
        self.saver=tf.train.Saver()
        
        #session
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        
        
    def Q_vals(self,state):
        """state:現在の盤面
        　　各ステップ終了後の盤面のこと"""
        feed={self.x:[state]}
        return self.sess.run(self.y,feed_dict=feed)[0]
    
    def epsilon_greedy(self,state,epsilon,targets):
        """targetsは行動
           epsilon_greedy()は行動を返す関数"""
        if np.random.rand()<epsilon:
            targets_new=[]
            for i in range(len(targets)):
                targets_new.append(self.axis2index(targets[i]))
                
            act=np.random.choice(targets_new)
            return act
        else:
            #Qを最大にするような行動をとる
            qvalue,action=self.next_able_action(state,targets)
            return action
        
        
        
        
    def next_able_action(self,state,targets):
        Q=self.Q_vals(state)
        
        index=np.argsort(Q)
        targets_new=[]
        for i in range(len(targets)):
            targets_new.append(self.axis2index(targets[i]))
        
        for action in reversed(index):
            if action in targets_new:
                break
                
        qvalue=Q[action]
        
        return qvalue,action
    
    #経験を学んでいく関数
    #遷移Dに状態、行動、報酬、終了判定を保存
    def store_experience(self,state,targets,action,reward,state_1,targets_1,terminal):
        """terminalは終了判定"""
        self.D.append((state,targets,action,reward,state_1,targets_1,terminal))
        
    #indexとaxisを行ったり来たりする    
    def axis2index(self,action):
        index=action[0]*8+action[1]
        return index
    
    def index2axis(self,index):
        r=int(index/8)
        c=index%8
        return [r,c]
        
    #経験を再現して学習する関数
    #遷移Dからミニバッチ的に取り出してくる
    def experience_replay(self):
        #minibatch learningの準備
        state_minibatch=[]
        y_minibatch=[]
        
        #Dのstateの数がミニバッチよりも小さかった時は
        minibatch_size=min(len(self.D),self.batch_size)
        indexes=np.random.randint(0,len(self.D),minibatch_size)
        
        for j in indexes:
            #遷移Dから取り出す
            state_j,targets_j,action_j,reward_j,state_j_1,targets_j_1,terminal=self.D[j]
            #行動にインデックスをつける
            action_j=self.axis2index(action_j)
            action_j_index=self.enable_actions.index(action_j)
            #価値関数、教師信号の初期化
            y_j=self.Q_vals(state_j)
            
            if terminal:
                y_j[action_j_index]=reward_j
            else:
                #next_able_action()でQが最大化した値と行動がとってこれる
                qvalue,action=self.next_able_action(state_j_1,targets_j_1)
                y_j[action_j_index]=reward_j+self.discount_factor*qvalue
                
                
            state_minibatch.append(state_j)
            y_minibatch.append(y_j)
            
            
            
        #training
        """入力データはstate_minibatch
           教師データはy_minibatch"""
        self.sess.run(self.optimizer,feed_dict={self.x:state_minibatch,self.y_:y_minibatch})
        
        #log
        self.current_loss=self.sess.run(self.loss,feed_dict={self.x:state_minibatch,self.y_:y_minibatch})
        
        
    #load,saveする関数
    def save_model(self,epoch,path='./osero-model/'):
        if not os.path.isdir(path):
            os.mkdir(path)
            
        self.saver.save(self.sess,os.path.join(path,'model.ckpt'),global_step=epoch)
        
    def load_model(self,epoch,path=None):
        if path:
            self.saver.restore(self.sess,path)
        else:
            self.saver.restore(self.sess,os.path.join(path,'model.ckpt-%d'%epoch))
                
                
    
        
        
                
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

