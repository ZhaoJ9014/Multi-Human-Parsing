import os
import tensorflow as tf 

def add_prefix(pref,foldername,save_path):
	for var_name, _ in tf.contrib.framework.list_variables(foldername):
		var = tf.contrib.framework.load_variable(foldername,var_name)
		if 'Adam' in var_name:
			var_name = pref + var_name
		new_name = pref + var_name
		var = tf.Variable(var,name=new_name)
		print('%s\t%s'%(var_name,new_name))
		print(var)

	with tf.Session() as sess:
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		saver.save(sess,save_path)

if not os.path.exists('./savings_bgfg/'):
	os.mkdir('./savings_bgfg/')

if not os.path.exists('./savings_seg/'):
	os.mkdir('./savings_seg/')

if not os.path.exists('./savings_inst_model/'):
	os.mkdir('./savings_inst_model/')

add_prefix('bg_fg/','./model/','./savings_bgfg/pretrain.ckpt')
add_prefix('seg_part/','./model/','./savings_seg/pretrain.ckpt')
add_prefix('inst_part/','./model/','./savings_inst_model/pretrain.ckpt')
