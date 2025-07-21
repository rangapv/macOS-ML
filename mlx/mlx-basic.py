#!/usr/bin/env python3
#author:rangapv@yahoo.com
#21-07-25

import mlx.core as mx

s1 = mx.metal.is_available() 
s2= mx.metal.device_info()

print(f'core is {s1}')
print(f'core is {s2}')
