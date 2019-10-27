from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm

'''
how to install reportlab
https://www.reportlab.com/dev/install/version_3_and_up/
pip install rlextra -i https://www.reportlab.com/pypi/
'''
#stworzy PDF o takiej nazwie
c = canvas.Canvas('ex.pdf')

c.setLineWidth(.3)
c.setFont('Helvetica', 12)

c.drawString(30, 750, 'OFFICIAL COMMUNIQUE')
c.drawString(30, 735, 'OF ACME INDUSTRIES')
c.drawString(500, 750, "12/12/2010")
c.line(480, 747, 580, 747)

#wstawia takie zdjęcie
c.drawImage('0A36B673BA6513F772FB78FF597BE44F7E639A0F.jpg', 5*cm, 0, 10*cm, 10*cm)
c.showPage()
c.save()