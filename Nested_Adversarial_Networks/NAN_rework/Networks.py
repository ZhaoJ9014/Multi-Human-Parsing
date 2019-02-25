import layers2 as L 
import modeleag as M 
import tensorflow as tf 

class ConvBlocks(M.Model):
	def initialize(self, fmap, stride=2):
		self.bn0 = L.batch_norm()
		self.activ = L.activation(M.PARAM_RELU)
		self.c1 = L.conv2D(3, fmap, stride=stride, pad='VALID', usebias=False)
		self.bn1 = L.batch_norm()
		self.c2 = L.conv2D(3, fmap, pad='VALID', usebias=False)
		# shortcut conv
		self.c3 = L.conv2D(1, fmap, stride=stride, usebias=False)

	def forward(self, x):
		x = self.bn0(x)
		x = self.activ(x)

		short = self.c3(x)

		branch = self.c1(M.pad(x, 1))
		branch = self.activ(self.bn1(branch))
		branch = self.c2(M.pad(branch, 1))

		res = branch + short
		return res 

class IdentityBlock(M.Model):
	def initialize(self, fmap):
		self.bn0 = L.batch_norm()
		self.activ = L.activation(M.PARAM_RELU)
		self.c1 = L.conv2D(3, fmap, pad='VALID', usebias=False)
		self.bn1 = L.batch_norm()
		self.c2 = L.conv2D(3, fmap, pad='VALID', usebias=False)

	def forward(self, x):
		short = x

		x = self.bn0(x)
		x = self.activ(x)
		branch = self.c1(M.pad(x, 1))
		branch = self.activ(self.bn1(branch))
		branch = self.c2(M.pad(branch, 1))

		res = branch + short
		return res 

class Block5(M.Model):
	def initialize(self):
		self.bn0 = L.batch_norm()
		self.activ = L.activation(M.PARAM_RELU)
		self.c1 = L.conv2D(3, 512, pad='VALID', usebias=False)
		self.bn1 = L.batch_norm()
		self.c2 = L.conv2D(3, 1024, pad='VALID', usebias=False, dilation_rate=2)
		# shortcut conv
		self.c3 = L.conv2D(1, 1024, usebias=False)

	def forward(self, x):
		x = self.bn0(x)
		x = self.activ(x)

		short = self.c3(x)

		branch = self.c1(M.pad(x, 1))
		branch = self.activ(self.bn1(branch))
		branch = self.c2(M.pad(branch, 2))

		res = branch + short
		return res 

class ID5Block(M.Model):
	def initialize(self):
		self.bn0 = L.batch_norm()
		self.activ = L.activation(M.PARAM_RELU)
		self.c1 = L.conv2D(3, 512, dilation_rate=2, pad='VALID', usebias=False)
		self.bn1 = L.batch_norm()
		self.c2 = L.conv2D(3, 1024, dilation_rate=2, pad='VALID', usebias=False)

	def forward(self, x):
		short = x

		x = self.bn0(x)
		x = self.activ(x)
		branch = self.c1(M.pad(x, 2))
		branch = self.activ(self.bn1(branch))
		branch = self.c2(M.pad(branch, 2))

		res = branch + short
		return res 

class LastBlock(M.Model):
	def initialize(self, fmaps, dilation_rate):
		self.dilation_rate = dilation_rate
		self.bn0 = L.batch_norm()
		self.activ = L.activation(M.PARAM_RELU)
		self.c1 = L.conv2D(1, fmaps[0], pad='VALID', usebias=False)
		self.bn1 = L.batch_norm()
		self.c2 = L.conv2D(3, fmaps[1], pad='VALID', usebias=False, dilation_rate=dilation_rate)
		self.bn2 = L.batch_norm()
		self.c3 = L.conv2D(1, fmaps[2], pad='VALID', usebias=False)

		# shortcut
		self.c4 = L.conv2D(1, fmaps[2], pad='VALID', usebias=False)
	def forward(self, x):
		x = self.bn0(x)
		x = self.activ(x)

		short = self.c4(x)

		branch = self.activ(self.bn1(self.c1(x)))
		branch = self.activ(self.bn2(self.c2(M.pad(branch, self.dilation_rate))))
		branch = self.c3(branch)

		res = branch + short
		return res 

class MainBody(M.Model):
	def initialize(self):
		self.c0 = L.conv2D(3, 64, usebias=False)
		# c0
		self.r1_0 = ConvBlocks(128)
		self.r1_1 = IdentityBlock(128)
		self.r1_2 = IdentityBlock(128)
		# c1
		self.r2_0 = ConvBlocks(256)
		self.r2_1 = IdentityBlock(256)
		self.r2_2 = IdentityBlock(256)
		# c2 
		self.r3_0 = ConvBlocks(256)
		self.r3_1 = IdentityBlock(256)
		self.r3_2 = IdentityBlock(256)
		self.r3_3 = IdentityBlock(256)
		self.r3_4 = IdentityBlock(256)
		self.r3_5 = IdentityBlock(256)
		# c3
		self.r4_0 = Block5()
		self.r4_1 = ID5Block()
		self.r4_2 = ID5Block()
		self.r4_3 = LastBlock([512,1024,2048], 4)
		self.r4_4 = LastBlock([1024, 2048, 4096], 4)
		self.bn4 = L.batch_norm()
		self.c4 = L.conv2D(3, 512, dilation_rate=12, pad='VALID')

	def forward(self,x):
		c0 = x = self.c0(x)
		x = self.r1_0(x)
		x = self.r1_1(x)
		c1 = x = self.r1_2(x)
		x = self.r2_0(x)
		x = self.r2_1(x)
		c2 = x = self.r2_2(x)
		x = self.r3_0(x)
		x = self.r3_1(x)
		x = self.r3_2(x)
		x = self.r3_3(x)
		x = self.r3_4(x)
		c3 = x = self.r3_5(x)
		x = self.r4_0(x)
		x = self.r4_1(x)
		x = self.r4_2(x)
		x = self.r4_3(x)
		x = self.r4_4(x)
		x = self.bn4(x)
		x = tf.nn.relu(x)
		x = M.pad(x, 12)
		x = self.c4(x)
		c4 = x = tf.nn.relu(x)
		return c0,c1,c2,c3,c4

class BGFGNet(M.Model):
	def initialize(self, class_num, dilation_rate):
		self.body = MainBody()
		self.seg_layer = L.conv2D(3, class_num, dilation_rate=dilation_rate)

	def forward(self, x):
		features = self.body(x)
		res = self.seg_layer(features[-1])
		# original tf.image.resize function sucks, we need to use our own function
		res = L.bilinear_upsample(res, 4)
		return res 

class SemSegNet(M.Model):
	def initialize(self, class_num, dilation_rate):
		self.merging = L.conv2D(1, 3, usebias=False)
		self.body = MainBody()
		self.seg_layer = L.conv2D(3, class_num, dilation_rate=dilation_rate)

	def forward(self, x, bgfg):
		x = tf.concat([x, bgfg], axis=-1)
		x = self.merging(x)
		features = self.body(x)
		res = self.seg_layer(features[-1])
		# original tf.image.resize function sucks, we need to use our own function
		res = L.bilinear_upsample(res, 4)
		return res 

class InstSegStream(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(3, 128, activation=M.PARAM_RELU)
		self.c2 = M.ConvLayer(3, 128, activation=M.PARAM_RELU)
		self.c3 = M.ConvLayer(3, 6)

	def forward(self, x):
		return self.c3(self.c2(self.c1(x)))

class InstSegNet(M.Model):
	def initialize(self):
		self.merging = L.conv2D(1, 3, usebias=False)
		self.body = MainBody()
		self.streams = [InstSegStream() for _ in range(4)]
		self.streams_fusion = L.conv2D(1, 6, usebias=False)

	def forward(self, x, bgfg, seg):
		x = tf.concat([x, bgfg, seg], axis=-1)
		x = self.merging(x)
		features = self.body(x)

		# merge inst pred layers
		stream_outputs = []
		features = features[1:]
		sample_rate = [2,4,8,8]
		for i,f in enumerate(features):
			stream = self.streams[i](f)
			stream = M.bilinear_upsample(stream, sample_rate[i])
			stream_outputs.append(stream)

		streams_fusion = self.streams_fusion(tf.concat(stream_outputs, -1))
		stream_outputs.append(streams_fusion)
		return stream_outputs

class DisNet(M.Model):
	def initialize(self):
		self.c0 = M.ConvLayer(5, 16, stride=1, activation=M.PARAM_RELU)
		self.c1 = M.ConvLayer(5, 32, stride=2, activation=M.PARAM_RELU) 
		self.c2 = M.ConvLayer(5, 64, stride=1, activation=M.PARAM_RELU)
		self.c3 = M.ConvLayer(5, 128, stride=2, activation=M.PARAM_RELU)
		self.c4 = M.ConvLayer(3, 256, stride=2, activation=M.PARAM_RELU)
		self.c5 = M.ConvLayer(1, 1)

	def forward(self, x):
		logits = self.c5(self.c4(self.c3(self.c2(self.c1(self.c0(x))))))
		return logits 
