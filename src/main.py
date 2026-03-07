import scripts.embed as emb
import scripts.inference as inf
import torch
import smtplib

device = 'cpu'

from scripts.core.model import MobileFacenet
from scrfd import SCRFD, Threshold
from email.message import EmailMessage
import ssl 

# import models
mfn_model = MobileFacenet().to(device)
checkpoint = torch.load('models/mobile_face_net.ckpt', map_location=device)
mfn_model.load_state_dict(checkpoint['net_state_dict'])
mfn_model.eval()

scrfd_model = SCRFD.from_path('models/scrfd.onnx')
print('SCRFD loaded')

print('------WELCOME TO P.A.T------')
depname = input('Enter department name (e.g., BCOE): ')
coursename = input('Enter course name (e.g., CS131): ')

department = depname
course = coursename

decision = ' '

while (decision != 'quit'):
    decision = input('1. Add student to database\n2.Take Attendance \n3.Display Registered Students \n4.Send Attendance Sheet to Google Drive' \
    '\nType "quit" to exit\nEnter your choice: ')

    if decision == '1':
        # ## add student to database
        emb.live_capture_faces(dir_storage=f'data/{department}.db',
                               course_section=course,
                               scrfd_model=scrfd_model,
                               mfn_model=mfn_model)
        
    elif decision == '2':
         # # train classifier (this specifies which course it will be take attendance)
        knn, le = emb.train_knn(dir_storage=f'data/{department}.db',course_section=course)
        inf.enable_inference(scrfd_model, mfn_model, knn, le, thresh=0.7)

    elif decision == '3':
        #Placeholder for print department.db
        print('Registered students:')


    elif decision == '4':
        # --- Email Configuration ---
        sender_email = input('Enter your email address: ')
        receiver_email = input('Enter the recipient email address: ')
        # Use the generated App Password, NOT your regular password
        app_password = input('Enter your email app password: ')

        # --- Create the email message ---
        msg = EmailMessage()
        msg.set_content('Please find the attached database file for attendance.') 
        # Inside elif decision == '4':
        
        file_path = f'data/{department}.db'
        with open(file_path, 'rb') as f:
            file_data = f.read()
            msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=f'{department}.db') 

        msg['Subject'] = "Database file for attendance"
        msg['From'] = sender_email
        msg['To'] = receiver_email

        # --- Send the email ---
        try:
            # Create a secure SSL context
            context = ssl.create_default_context() 
            
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, app_password)
                server.send_message(msg)
            print("Email sent successfully!")
        except Exception as e:
            print(f"Error sending email: {e}")

    elif decision == 'quit' or decision == 'exit' or decision == 'q':
        print('Exiting the program. Goodbye!')
        break

    else:
        print('Invalid choice. Please try again.')
        continue







# ## add student to database
# emb.live_capture_faces(dir_storage=f'data/{department}.db',
#                        course_section=course,
#                        scrfd_model=scrfd_model,
#                        mfn_model=mfn_model)


# # train classifier (this specifies which course it will be take attendance)
# knn, le = emb.train_knn(dir_storage=f'data/{department}.db',
#                         course_section=course)


# # take attendance
# inf.enable_inference(scrfd_model, mfn_model, knn, le, thresh=0.7)





