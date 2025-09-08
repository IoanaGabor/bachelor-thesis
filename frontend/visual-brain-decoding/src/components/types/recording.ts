export interface Recording {
    id: string;
    nifti_file: File;
    png_file: File;
    description?: string;
    createdAt: string;
}
  