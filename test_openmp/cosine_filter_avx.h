void cosine_filter_avx(
	float* fr_data, float* fb_array, 
	const int height, const int width, 
	const int filter_h, const int filter_w, 
	const int n_filters, float* out_data)
{
	transposeBank(fb_array);
	//do convolution
	const int apron_y = filter_h / 2;
	const int apron_x = filter_w / 2;

	const int filter_size = filter_h * filter_w;

	const int filter_bank_size = filter_size * n_filters;

	int *pixel_offsets=(int*) malloc(sizeof(int)*filter_size);

	int oi = 0;
	for (int ii=-apron_y; ii<=apron_y; ii++){
		for (int jj=-apron_y; jj<=apron_y; jj++){
			pixel_offsets[oi] = ii * width + jj;
			oi++;
		}
	}
	// 100 filters, each 9 values
	int imask = 0x7fffffff;
	float fmask = *((float*)&imask);

	int n_threads = omp_get_num_procs();
	omp_set_num_threads(n_threads);

	double tic = omp_get_wtime();
	for(int i=0; i<1000; i++) 
	{

		int valid_height = height - 2 * apron_y;
		int height_step = valid_height / n_threads + 1;

		#pragma omp parallel for
		for (int tid=0; tid<n_threads; tid++){
			int start_y = apron_y + tid * height_step;
			int end_y = min(start_y + height_step, height - apron_y);
	//		for(int i=0; i<100; i++)
				for (int i=start_y; i<end_y; i++){
					float* fr_ptr = fr_data + i * width + apron_x;
					float* ass_out = out_data + i * width + apron_x;
					float* wgt_out = ass_out + height * width;

					for (int j=apron_x; j<(width - apron_x); j++ ){


						__m256 image_cache0 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[0]]);
						__m256 image_cache1 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[1]]);
						__m256 image_cache2 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[2]]);
						__m256 image_cache3 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[3]]);
						__m256 image_cache4 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[4]]);
						__m256 image_cache5 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[5]]);
						__m256 image_cache6 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[6]]);
						__m256 image_cache7 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[7]]);
						__m256 image_cache8 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[8]]);

						float max_sim[8] = {-1e6, 
							-1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6};
						int best_ind = -1;

						int fi=0;
						int filter_ind = 0;

						// 96 filters, 9 values each
						while (fi<((n_filters/8)*8)*filter_size)
						{
							__m256 temp_sum = _mm256_set1_ps(0.0f);

							// no fused multiply add :(
							// current value of 8 filters
							__m256 curr_filter = _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache0, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache1, curr_filter), temp_sum);

							curr_filter = _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache2, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache3, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache4, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache5, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache6, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache7, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache8, curr_filter), temp_sum);

							// calculating absolute value by clearing the last digit
							__m256 mask = _mm256_set1_ps(fmask);

							temp_sum = _mm256_and_ps(mask, temp_sum);

							__m256 max_fil = _mm256_load_ps(max_sim);
							// code 14

							int r;

							// low 128 half
							// copy low to high
							__m256 temp_sum2 = _mm256_insertf128_ps(temp_sum,
								_mm256_extractf128_ps(temp_sum, 0), 1);
							__m256 cpm = _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS);
							r = _mm256_movemask_ps(cpm);


							if(r&(1<<0)) {
								best_ind = filter_ind+7;
								int control = 0;
								max_fil = _mm256_permute_ps(temp_sum2, control); 
								r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
							}

							if(r&(1<<1)) {
								best_ind = filter_ind+6;
								int control = 1|(1<<2)|(1<<4)|(1<<6);
								max_fil = _mm256_permute_ps(temp_sum2, control); 
								r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
							}

							if(r&(1<<2)) {
								best_ind = filter_ind+5;
								int control = 2|(2<<2)|(2<<4)|(2<<6);
								max_fil = _mm256_permute_ps(temp_sum2, control); 
								r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
							}

							if(r&(1<<3)) {
								best_ind = filter_ind+4;
								int control = 3|(3<<2)|(3<<4)|(3<<6);
								max_fil = _mm256_permute_ps(temp_sum2, control); 
								r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
							}

							// high 128 half
							// copy high to low
							temp_sum2 = _mm256_insertf128_ps(temp_sum,
								_mm256_extractf128_ps(temp_sum, 1), 0);
							cpm = _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS);
							r = _mm256_movemask_ps(cpm);


							if(r&(1<<0)) {
								best_ind = filter_ind+3;
								int control = 0;
								max_fil = _mm256_permute_ps(temp_sum2, control); 
								r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
							}

							if(r&(1<<1)) {
								best_ind = filter_ind+2;
								int control = 1|(1<<2)|(1<<4)|(1<<6);
								max_fil = _mm256_permute_ps(temp_sum2, control); 
								r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
							}

							if(r&(1<<2)) {
								best_ind = filter_ind+1;
								int control = 2|(2<<2)|(2<<4)|(2<<6);
								max_fil = _mm256_permute_ps(temp_sum2, control); 
								r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
							}

							if(r&(1<<3)) {
								best_ind = filter_ind+0;
								int control = 3|(3<<2)|(3<<4)|(3<<6);
								max_fil = _mm256_permute_ps(temp_sum2, control); 
								r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
							}


							_mm256_store_ps(max_sim, max_fil);
							// printf("max1 :%f\n", max_fil.m128_f32[0]);


							filter_ind += 8;
						}

						// leftover filters
						__m128 temp_sum = _mm_set1_ps(0.0f);


						// current value of 4 filters
						__m128 curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache0, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache1, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache2, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache3, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache4, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache5, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache6, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache7, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache8, 0), curr_filter), temp_sum);

						__m128 max_fil = _mm_load_ss(max_sim);

						__m128 cpm = _mm_cmp_ps(temp_sum, max_fil, _CMP_GT_OS);
						int	r = _mm_movemask_ps(cpm);


						if(r&(1<<0)) {
							best_ind = filter_ind+3;
							int control = 0;
							max_fil = _mm_permute_ps(temp_sum, control); 
							r=_mm_movemask_ps( _mm_cmp_ps(temp_sum, max_fil, _CMP_GT_OS));
						}

						if(r&(1<<1)) {
							best_ind = filter_ind+2;
							int control = 1|(1<<2)|(1<<4)|(1<<6);
							max_fil = _mm_permute_ps(temp_sum, control); 
							r=_mm_movemask_ps( _mm_cmp_ps(temp_sum, max_fil, _CMP_GT_OS));
						}

						if(r&(1<<2)) {
							best_ind = filter_ind+1;
							int control = 2|(2<<2)|(2<<4)|(2<<6);
							max_fil = _mm_permute_ps(temp_sum, control); 
							r=_mm_movemask_ps( _mm_cmp_ps(temp_sum, max_fil, _CMP_GT_OS));
						}

						if(r&(1<<3)) {
							best_ind = filter_ind+0;
							int control = 3|(3<<2)|(3<<4)|(3<<6);
							max_fil = _mm_permute_ps(temp_sum, control); 
							r=_mm_movemask_ps( _mm_cmp_ps(temp_sum, max_fil, _CMP_GT_OS));
						}
						_mm_store_ps(max_sim, max_fil);

						*ass_out = (float)best_ind;
						*wgt_out = max_sim[0];

						fr_ptr++;
						ass_out++;
						wgt_out++;
					}
				}
		}
		//#pragma omp barrier
	}
	double toc = omp_get_wtime();
	std::cout << "openmp filter avx time: " << toc - tic << std::endl;
}


void cosine_filter_avx_nocomp(
	float* fr_data, float* fb_array, 
	const int height, const int width, 
	const int filter_h, const int filter_w, 
	const int n_filters, float* out_data)
{
	transposeBank(fb_array);
	//do convolution
	const int apron_y = filter_h / 2;
	const int apron_x = filter_w / 2;

	const int filter_size = filter_h * filter_w;

	const int filter_bank_size = filter_size * n_filters;

	int *pixel_offsets=(int*) malloc(sizeof(int)*filter_size);

	int oi = 0;
	for (int ii=-apron_y; ii<=apron_y; ii++){
		for (int jj=-apron_y; jj<=apron_y; jj++){
			pixel_offsets[oi] = ii * width + jj;
			oi++;
		}
	}
	// 100 filters, each 9 values
	int imask = 0x7fffffff;
	float fmask = *((float*)&imask);

	int n_threads = omp_get_num_procs();
	omp_set_num_threads(n_threads);

	double tic = omp_get_wtime();
	for(int i=0; i<1000; i++) 
	{

		int valid_height = height - 2 * apron_y;
		int height_step = valid_height / n_threads + 1;

		#pragma omp parallel for
		for (int tid=0; tid<n_threads; tid++){
			int start_y = apron_y + tid * height_step;
			int end_y = min(start_y + height_step, height - apron_y);
	//		for(int i=0; i<100; i++)
				for (int i=start_y; i<end_y; i++){
					float* fr_ptr = fr_data + i * width + apron_x;
					float* ass_out = out_data + i * width + apron_x;
					float* wgt_out = ass_out + height * width;

					for (int j=apron_x; j<(width - apron_x); j++ ){


						__m256 image_cache0 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[0]]);
						__m256 image_cache1 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[1]]);
						__m256 image_cache2 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[2]]);
						__m256 image_cache3 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[3]]);
						__m256 image_cache4 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[4]]);
						__m256 image_cache5 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[5]]);
						__m256 image_cache6 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[6]]);
						__m256 image_cache7 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[7]]);
						__m256 image_cache8 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[8]]);

						__declspec(align(64)) float max_sim[8] = {-1e6, 
							-1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6};
						int best_ind = -1;

						int fi=0;
						int filter_ind = 0;

						// 96 filters, 9 values each
						while (fi<((n_filters/8)*8)*filter_size)
						{
							__m256 temp_sum = _mm256_set1_ps(0.0f);

							// no fused multiply add :(
							// current value of 8 filters
							__m256 curr_filter = _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache0, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache1, curr_filter), temp_sum);

							curr_filter = _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache2, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache3, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache4, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache5, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache6, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache7, curr_filter), temp_sum);

							curr_filter= _mm256_load_ps(&fb_array[fi]);
							fi+=8;
							temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache8, curr_filter), temp_sum);

							// calculating absolute value by clearing the last digit
							__m256 mask = _mm256_set1_ps(fmask);

							temp_sum = _mm256_and_ps(mask, temp_sum);

							__m256 max_fil = _mm256_load_ps(max_sim);
							// code 14
							max_fil =  _mm256_hadd_ps(temp_sum, max_fil);
						/*	max_fil = _mm256_max_ps(max_fil, temp_sum);
							__m256 max1 =  _mm256_castps128_ps256(_mm256_extractf128_ps(max_fil, 0));
							__m256 max2 =  _mm256_castps128_ps256(_mm256_extractf128_ps(max_fil, 1));
							// compare upper and lower half
							max_fil = _mm256_max_ps(max1, max2);
							// move 3rd and 4th to 1st and 2nd positions
							int control = 2|(3<<2);
							max_fil = _mm256_max_ps(max_fil, _mm256_permute_ps(max_fil, control));
							control = 1;
							max_fil = _mm256_max_ps(max_fil, _mm256_permute_ps(max_fil, control));
							max_fil = _mm256_insertf128_ps(max_fil,
								_mm256_extractf128_ps(max_fil, 0), 1);
							max_fil = _mm256_permute_ps(max_fil, 0);
							*/

							_mm256_store_ps(max_sim, max_fil);
							// printf("max1 :%f\n", max_fil.m128_f32[0]);


							filter_ind += 8;
						}

						// leftover filters
						__m128 temp_sum = _mm_set1_ps(0.0f);


						// current value of 4 filters
						__m128 curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache0, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache1, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache2, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache3, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache4, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache5, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache6, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache7, 0), curr_filter), temp_sum);

						curr_filter = _mm_load_ps(&fb_array[fi]);
						fi+=4;
						temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache8, 0), curr_filter), temp_sum);

						
						__m256 max_fil = _mm256_load_ps(max_sim);
						max_fil = _mm256_max_ps(_mm256_castps128_ps256(temp_sum), max_fil);
						
						_mm256_store_ps(max_sim, max_fil);

						*ass_out = (float)best_ind;
						*wgt_out = max_sim[0];

						fr_ptr++;
						ass_out++;
						wgt_out++;
					}
				}
		}
		//#pragma omp barrier
	}
	double toc = omp_get_wtime();
	std::cout << "openmp filter avx no comp time: " << toc - tic << std::endl;
}
