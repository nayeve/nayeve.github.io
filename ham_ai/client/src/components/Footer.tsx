import { ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-slate-800 text-white py-12">
      <div className="max-w-7xl mx-auto px-4">
        <div className="grid md:grid-cols-4 gap-8 mb-8">
          <div>
            <h3 className="text-lg font-semibold mb-4">Official Links</h3>
            <ul className="space-y-2 text-sm">
              <li><a href="https://usa.gov" className="hover:text-blue-300 flex items-center" target="_blank" rel="noopener noreferrer">USA.gov <ExternalLink className="h-3 w-3 ml-1" /></a></li>
              <li><a href="https://jaycorp.gov" className="hover:text-blue-300 flex items-center" target="_blank" rel="noopener noreferrer">Jaycorp Communication Commission <ExternalLink className="h-3 w-3 ml-1" /></a></li>
              <li><a href="https://nist.gov" className="hover:text-blue-300 flex items-center" target="_blank" rel="noopener noreferrer">NIST.gov <ExternalLink className="h-3 w-3 ml-1" /></a></li>
              <li><a href="https://arrl.org" className="hover:text-blue-300 flex items-center" target="_blank" rel="noopener noreferrer">ARRL.org <ExternalLink className="h-3 w-3 ml-1" /></a></li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Resources</h3>
            <ul className="space-y-2 text-sm">
              <li><a href="#applications" className="hover:text-blue-300">Implementation Guides</a></li>
              <li><a href="#sources" className="hover:text-blue-300">Technical Documentation</a></li>
              <li><a href="#" className="hover:text-blue-300">Training Materials</a></li>
              <li><a href="#" className="hover:text-blue-300">Research Papers</a></li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Support</h3>
            <ul className="space-y-2 text-sm">
              <li><a href="#author" className="hover:text-blue-300">Contact Information</a></li>
              <li><a href="#" className="hover:text-blue-300">Technical Support</a></li>
              <li><a href="#" className="hover:text-blue-300">Community Forums</a></li>
              <li><a href="#" className="hover:text-blue-300">Feedback</a></li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Legal</h3>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="hover:text-blue-300">Privacy Policy</a></li>
              <li><a href="#" className="hover:text-blue-300">Terms of Use</a></li>
              <li><a href="#" className="hover:text-blue-300">Accessibility</a></li>
              <li><a href="#" className="hover:text-blue-300">No FEAR Act</a></li>
            </ul>
          </div>
        </div>
        
        <div className="border-t border-gray-600 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center text-sm">
            <div className="mb-4 md:mb-0">
              <p>&copy; 2024 Jay States Communications Authority. All rights reserved.</p>
              <p>An official website of the Jay States Communications Authority</p>
            </div>
            <div className="flex items-center space-x-4">
              <span className="flex items-center">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                Secure
              </span>
              <span className="flex items-center">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                Encrypted
              </span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
