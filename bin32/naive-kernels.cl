// Dla adresowania uint8*
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uchar erodeINF = 255;
__constant uchar dilateINF = 0;
__constant uchar OBJ = 255;
__constant uchar BCK = 0;

__kernel void subtract(
	__global uchar* a,
	__global uchar* b,
	__global uchar* output)
{
	size_t gid = get_global_id(0);
	output[gid] = (b[gid] > a[gid]) ? (0): (a[gid] - b[gid]);
}

__kernel void addHalf(
	__global uchar* skeleton,
	__global uchar* src)
{
	size_t gid = get_global_id(0);
	
	uchar v = skeleton[gid];
	if(v == 0)
	{
		skeleton[gid] = src[gid] / 2;
	}
}

__kernel void erode(
	__global uchar* input,
	__global uchar* output,
	__constant uchar* element,
	const int anchorX,
	const int anchorY)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	
	size_t gwidth = get_global_size(0);
	size_t gheight = get_global_size(1);
	
	uchar val = erodeINF;
	int eid = 0;

	for(int y = -1 * anchorY; y < (anchorY + 1); ++y)
	{
		for(int x = -1 * anchorX; x < (anchorX + 1); ++x)
		{
			int xi = gid.x + x;
			int yi = gid.y + y;
			
			if(xi < 0 || xi >= gwidth || yi < 0 || yi >= gheight)
			{
				if(element[eid] != 0)
					val = min(val, erodeINF);
			}
			
			else if(element[eid] != 0)
			{
				val = min(val, input[xi + yi * gwidth]);
			}
			
			++eid;
		}
	}
		
	output[gid.x + gid.y * gwidth] = val;
}

__kernel void dilate(
	__global uchar* input,
	__global uchar* output,
	__constant uchar* element,
	const int anchorX,
	const int anchorY)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	
	size_t gwidth = get_global_size(0);
	size_t gheight = get_global_size(1);
	
	uchar val = dilateINF;
	int eid = 0;

	for(int y = -1 * anchorY; y < (anchorY + 1); ++y)
	{
		for(int x = -1 * anchorX; x < (anchorX + 1); ++x)
		{
			int xi = gid.x + x;
			int yi = gid.y + y;
			
			if(xi < 0 || xi >= gwidth || yi < 0 || yi >= gheight)
			{
				if(element[eid] != 0)
					val = max(val, dilateINF);
			}
			
			else if(element[eid] != 0)
			{
				val = max(val, input[xi + yi * gwidth]);
			}
			
			++eid;
		}
	}
		
	output[gid.x + gid.y * gwidth] = val;
}

__kernel void remove(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t gwidth = get_global_size(0);
	size_t gheight = get_global_size(0);
	
	if(gid.x == 0 || gid.x >= gwidth-1 || gid.y == 0 || gid.y >= gheight-1)
		return;
		
	if (input[(gid.x - 1) + (gid.y - 1) * gwidth] == OBJ &&
		input[(gid.x    ) + (gid.y - 1) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y - 1) * gwidth] == OBJ &&
		
		input[(gid.x - 1) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * gwidth] == OBJ &&
		
		input[(gid.x - 1) + (gid.y + 1) * gwidth] == OBJ &&
		input[(gid.x    ) + (gid.y + 1) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y + 1) * gwidth] == OBJ)
	{
		output[gid.x + gid.y * gwidth] = BCK;
	}

/* 	Minimalnie szybsze
	
	size_t stepsize = gwidth - 2;
	size_t id = (gid.x - 1) + (gid.y - 1) * gwidth;
	// gorny wiersz
	if (input[id] == OBJ)
	{
		++id;
		if (input[id] == OBJ)
		{
			++id;
			if (input[id] == OBJ)
			{
				id += stepsize;
				
				// srodkowy wiersz
				if (input[id] == OBJ)
				{
					id += 2; // przeskakujemy srodek
					if (input[id] == OBJ)
					{
						id += stepsize;
						
						// dolny wiersz
						if(input[id] == OBJ)
						{
							++id;
							if(input[id] == OBJ)
							{
								++id;
								if(input[id] == OBJ)
								{
									output[gid.x + gid.y * gwidth] = BCK;
								}
							}
						}
					}
				}
			}
		}
	}
*/
}

__kernel void skeleton_iter1(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t gwidth = get_global_size(0);
	size_t gheight = get_global_size(0);
	
	if(gid.x == 0 || gid.x >= gwidth-1 || gid.y == 0 || gid.y >= gheight-1)
		return;
		
	// Element strukturalny pierwszy
	// 1|1|1
	// X|1|X
	// 0|0|0
	//
		
	if (input[(gid.x - 1) + (gid.y - 1) * gwidth] == OBJ &&
		input[(gid.x    ) + (gid.y - 1) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y - 1) * gwidth] == OBJ &&
		
		input[(gid.x    ) + (gid.y    ) * gwidth] == OBJ &&
		
		input[(gid.x - 1) + (gid.y + 1) * gwidth] == BCK &&
		input[(gid.x    ) + (gid.y + 1) * gwidth] == BCK &&
		input[(gid.x + 1) + (gid.y + 1) * gwidth] == BCK)
	{
		output[gid.x + gid.y * gwidth] = BCK;
	}
}

__kernel void skeleton_iter2(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t gwidth = get_global_size(0);
	size_t gheight = get_global_size(0);
	
	if(gid.x == 0 || gid.x >= gwidth-1 || gid.y == 0 || gid.y >= gheight-1)
		return;
		
	// Element strukturalny pierwszy - 90 w lewo
	// 1|X|0
	// 1|1|0
	// 1|x|0
	//
	
	if (input[(gid.x - 1) + (gid.y - 1) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y - 1) * gwidth] == BCK &&
		
		input[(gid.x - 1) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x    ) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * gwidth] == BCK &&
		
		input[(gid.x - 1) + (gid.y + 1) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y + 1) * gwidth] == BCK)
	{
		output[gid.x + gid.y * gwidth] = BCK;
	}
}

__kernel void skeleton_iter3(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t gwidth = get_global_size(0);
	size_t gheight = get_global_size(0);
	
	if(gid.x == 0 || gid.x >= gwidth-1 || gid.y == 0 || gid.y >= gheight-1)
		return;	
		
	// Element strukturalny pierwszy - 180 w lewo
	// 0|0|0
	// X|1|X
	// 1|1|1
	//
		
	if (input[(gid.x - 1) + (gid.y - 1) * gwidth] == BCK &&
		input[(gid.x    ) + (gid.y - 1) * gwidth] == BCK &&
		input[(gid.x + 1) + (gid.y - 1) * gwidth] == BCK &&
		input[(gid.x    ) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x - 1) + (gid.y + 1) * gwidth] == OBJ &&
		input[(gid.x    ) + (gid.y + 1) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y + 1) * gwidth] == OBJ)
	{
		output[gid.x + gid.y * gwidth] = BCK;
	}
}

__kernel void skeleton_iter4(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t gwidth = get_global_size(0);
	size_t gheight = get_global_size(0);
	
	if(gid.x == 0 || gid.x >= gwidth-1 || gid.y == 0 || gid.y >= gheight-1)
		return;
		
	// Element strukturalny pierwszy - 270 w lewo
	// 0|X|1
	// 0|1|1
	// 0|X|1
	//
	
	if (input[(gid.x - 1) + (gid.y - 1) * gwidth] == BCK &&
		input[(gid.x + 1) + (gid.y - 1) * gwidth] == OBJ &&
		input[(gid.x - 1) + (gid.y    ) * gwidth] == BCK &&
		input[(gid.x    ) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x - 1) + (gid.y + 1) * gwidth] == BCK &&
		input[(gid.x + 1) + (gid.y + 1) * gwidth] == OBJ)
	{
		output[gid.x + gid.y * gwidth] = BCK;
	}
}

__kernel void skeleton_iter5(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t gwidth = get_global_size(0);
	size_t gheight = get_global_size(0);
	
	if(gid.x == 0 || gid.x >= gwidth-1 || gid.y == 0 || gid.y >= gheight-1)
		return;
		
	// Element strukturalny drugi
	// X|1|X
	// 0|1|1
	// 0|0|X
	//

	if (input[(gid.x    ) + (gid.y - 1) * gwidth] == OBJ &&
		input[(gid.x - 1) + (gid.y    ) * gwidth] == BCK &&
		input[(gid.x    ) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x - 1) + (gid.y + 1) * gwidth] == BCK &&
		input[(gid.x    ) + (gid.y + 1) * gwidth] == BCK)
	{
		output[gid.x + gid.y * gwidth] = BCK;
	}
}

__kernel void skeleton_iter6(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t gwidth = get_global_size(0);
	size_t gheight = get_global_size(0);
	
	if(gid.x == 0 || gid.x >= gwidth-1 || gid.y == 0 || gid.y >= gheight-1)
		return;
		
	// Element strukturalny drugi - 90 stopni w lewo
	// X|1|X
	// 1|1|0
	// X|0|0
	//
		
	if (input[(gid.x    ) + (gid.y - 1) * gwidth] == OBJ &&
		input[(gid.x - 1) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x    ) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * gwidth] == BCK &&
		input[(gid.x    ) + (gid.y + 1) * gwidth] == BCK &&
		input[(gid.x + 1) + (gid.y + 1) * gwidth] == BCK)
	{
		output[gid.x + gid.y * gwidth] = BCK;
	}
}

__kernel void skeleton_iter7(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t gwidth = get_global_size(0);
	size_t gheight = get_global_size(0);
	
	if(gid.x == 0 || gid.x >= gwidth-1 || gid.y == 0 || gid.y >= gheight-1)
		return;
		
	// Element strukturalny drugi - 180 stopni w lewo
	// X|0|0
	// 1|1|0
	// X|1|X
	//
		
	if (input[(gid.x    ) + (gid.y - 1) * gwidth] == BCK &&
		input[(gid.x + 1) + (gid.y - 1) * gwidth] == BCK &&
		input[(gid.x - 1) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x    ) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * gwidth] == BCK &&
		input[(gid.x    ) + (gid.y + 1) * gwidth] == OBJ)
	{
		output[gid.x + gid.y * gwidth] = BCK;
	}
}

__kernel void skeleton_iter8(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t gwidth = get_global_size(0);
	size_t gheight = get_global_size(0);
	
	if(gid.x == 0 || gid.x >= gwidth-1 || gid.y == 0 || gid.y >= gheight-1)
		return;
		
	// Element strukturalny drugi - 270 stopni w lewo
	// 0|0|X
	// 0|1|1
	// X|1|X
	//
		
	if (input[(gid.x - 1) + (gid.y - 1) * gwidth] == BCK &&
		input[(gid.x    ) + (gid.y - 1) * gwidth] == BCK &&
		input[(gid.x - 1) + (gid.y    ) * gwidth] == BCK &&
		input[(gid.x    ) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * gwidth] == OBJ &&
		input[(gid.x    ) + (gid.y + 1) * gwidth] == OBJ)
	{
		output[gid.x + gid.y * gwidth] = BCK;
	}
}