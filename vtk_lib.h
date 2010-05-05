
struct float2_
{
    float x;
    float y;
};
typedef struct float2_ float2;
void vtk_write_header(FILE *u,char const *title);
void vtk_write_grid_2d(FILE *f, int w, int h);
void vtk_write_vector_2d(FILE *f, char const *name, float2 * vectors, int len);
