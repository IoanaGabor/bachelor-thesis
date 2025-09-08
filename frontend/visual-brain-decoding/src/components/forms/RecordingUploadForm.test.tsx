import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import RecordingUploadForm from './RecordingUploadForm';

import { uploadRecording } from '../services/api-service';

import { ThemeProvider, createTheme } from '@mui/material/styles';

jest.mock('../services/api-service');

const theme = createTheme();

const renderWithTheme = (component: React.ReactElement) => {
  return render(<ThemeProvider theme={theme}>{component}</ThemeProvider>);
};

describe('RecordingUploadForm', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(window, 'alert').mockImplementation(() => {});
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('renders all form elements correctly', () => {
    renderWithTheme(<RecordingUploadForm />);

    expect(screen.getByLabelText(/fMRI voxel data file/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/PNG File/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Description \(optional\)/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Upload/i })).toBeInTheDocument();
  });

  test('handles description change', () => {
    renderWithTheme(<RecordingUploadForm />);
    const descriptionInput = screen.getByPlaceholderText(/Description \(optional\)/i);

    fireEvent.change(descriptionInput, { target: { value: 'Test description text.' } });

    expect(descriptionInput).toHaveValue('Test description text.');
  });

  test('shows error if npy file is missing on submission', async () => {
    renderWithTheme(<RecordingUploadForm />);
    const pngInput = screen.getByLabelText(/PNG File/i) as HTMLInputElement;
    const submitButton = screen.getByRole('button', { name: /Upload/i });

    const pngFile = new File(['png-content'], 'image.png', { type: 'image/png' });
    fireEvent.change(pngInput, { target: { files: [pngFile] } });

    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/Both files are required./i)).toBeInTheDocument();
    });
    expect(uploadRecording).not.toHaveBeenCalled();
  });

  test('shows error if PNG file is missing on submission', async () => {
    renderWithTheme(<RecordingUploadForm />);
    const niftiInput = screen.getByLabelText(/fMRI voxel data file/i) as HTMLInputElement;
    const submitButton = screen.getByRole('button', { name: /Upload/i });

    const niftiFile = new File(['nifti-content'], 'scan.nii.gz', { type: 'application/gzip' });
    fireEvent.change(niftiInput, { target: { files: [niftiFile] } });

    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/Both files are required./i)).toBeInTheDocument();
    });
    expect(uploadRecording).not.toHaveBeenCalled();
  });

  test('successfully uploads files and description', async () => {
    (uploadRecording as jest.Mock).mockResolvedValueOnce({ message: 'Success' });

    renderWithTheme(<RecordingUploadForm />);

    const niftiInput = screen.getByLabelText(/fMRI voxel data file/i) as HTMLInputElement;
    const pngInput = screen.getByLabelText(/PNG File/i) as HTMLInputElement;
    const descriptionInput = screen.getByPlaceholderText(/Description \(optional\)/i);
    const submitButton = screen.getByRole('button', { name: /Upload/i });

    const niftiFile = new File(['nifti-content'], 'scan.nii.gz', { type: 'application/gzip' });
    const pngFile = new File(['png-content'], 'image.png', { type: 'image/png' });

    fireEvent.change(niftiInput, { target: { files: [niftiFile] } });
    fireEvent.change(pngInput, { target: { files: [pngFile] } });
    fireEvent.change(descriptionInput, { target: { value: 'A test description.' } });

    fireEvent.click(submitButton);

    expect(submitButton).toHaveTextContent('Uploading...');
    expect(submitButton).toBeDisabled();

    await waitFor(() => {
      expect(uploadRecording).toHaveBeenCalledTimes(1);
    });

    const formDataArg = (uploadRecording as jest.Mock).mock.calls[0][0];
    expect(formDataArg instanceof FormData).toBe(true);
    expect(formDataArg.get('nifti_file')).toEqual(niftiFile);
    expect(formDataArg.get('png_file')).toEqual(pngFile);
    expect(formDataArg.get('description')).toBe('A test description.');

    expect(submitButton).toHaveTextContent('Upload');
    expect(submitButton).not.toBeDisabled();
  });

  test('displays error message on upload failure', async () => {
    (uploadRecording as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

    renderWithTheme(<RecordingUploadForm />);

    const niftiInput = screen.getByLabelText(/fMRI voxel data file/i) as HTMLInputElement;
    const pngInput = screen.getByLabelText(/PNG File/i) as HTMLInputElement;
    const submitButton = screen.getByRole('button', { name: /Upload/i });

    const niftiFile = new File(['nifti-content'], 'scan.nii.gz', { type: 'application/gzip' });
    const pngFile = new File(['png-content'], 'image.png', { type: 'image/png' });

    fireEvent.change(niftiInput, { target: { files: [niftiFile] } });
    fireEvent.change(pngInput, { target: { files: [pngFile] } });
    fireEvent.click(submitButton);

    expect(submitButton).toHaveTextContent('Uploading...');
    expect(submitButton).toBeDisabled();

    await waitFor(() => {
      expect(screen.getByText(/Upload failed./i)).toBeInTheDocument();
    });

    expect(console.error).toHaveBeenCalled();
    expect(uploadRecording).toHaveBeenCalledTimes(1);
    expect(submitButton).toHaveTextContent('Upload');
    expect(submitButton).not.toBeDisabled();
  });
});